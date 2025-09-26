import re, os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from perception_open_set_detector.classes.detection import Model_Detection, GroundTruthDetection
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
metric = util.cos_sim
# model = SentenceTransformer('all-MiniLM-L6-v2')

def create_detections(image, prediction, detection_boxes, custom_labels, threshold=0.1):
    matches = re.findall(r"<ground>(.*?)</ground><objects>(.*?)</objects>", prediction)
    temp_detections = [None] * len(detection_boxes)

    for label, obj_str in matches:

        if custom_labels:
        # Map label to best matching custom label or None if below threshold
            mapped_label = semantic_similarity(label.lower(), custom_labels, threshold)
            if mapped_label is None:
                continue
            label = mapped_label
        
        ids = re.findall(r"<obj(\d+)>", obj_str)
        for id_str in ids:
            idx = int(id_str)
            bbox = detection_boxes[idx]
            detection = Model_Detection(label=label, bbox=bbox)
            temp_detections[idx] = detection
            if image.coco_data: detection.to_coco(image)

    return [det for det in temp_detections if det is not None]

def get_all_files_recursive(base_dir, valid_extensions=None):
    if valid_extensions is None:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                full_path = os.path.join(root, file)
                all_files.append(full_path)
    return all_files

def load_and_merge_coco_annotations_all_subdirs(dataset_path, annotation_file="_annotations.coco.json") -> dict[str, list] :
    merged_coco = {"images": [], "annotations": [], "categories": []}
    image_id_offset = 0
    annotation_id_offset = 0
    categories_loaded = False

    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):
            continue

        annotation_path = os.path.join(subdir_path, annotation_file)
        if not os.path.isfile(annotation_path):
            print(f"Warning: Annotation file not found in {annotation_path}, skipping.")
            continue

        with open(annotation_path, "r") as f:
            coco_data = json.load(f)

        if not categories_loaded:
            merged_coco["categories"] = coco_data.get("categories", [])
            categories_loaded = True

        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])

        max_image_id = max([img["id"] for img in images], default=-1)
        max_ann_id = max([ann["id"] for ann in annotations], default=-1)

        for img in images:
            new_img = img.copy()
            new_img["id"] += image_id_offset
            merged_coco["images"].append(new_img)

        for ann in annotations:
            new_ann = ann.copy()
            new_ann["id"] += annotation_id_offset
            new_ann["image_id"] += image_id_offset
            merged_coco["annotations"].append(new_ann)

        image_id_offset += max_image_id + 1
        annotation_id_offset += max_ann_id + 1

    return merged_coco


def load_coco_annotations(dataset_path, annotation_file="_annotations.coco.json"):
    annotation_path = os.path.join(dataset_path, annotation_file)
    with open(annotation_path, "r") as f:
        coco_data = json.load(f)
    return coco_data, annotation_path

def semantic_similarity(sentence, list_of_sentences, threshold=0.5):
    max_value = 0
    max_sentence = sentence
    for key in list_of_sentences:
        value = metric(
            model.encode(sentence, show_progress_bar=False), 
            model.encode(key, show_progress_bar=False)
        ).item()
        if value > max_value:
            max_value = value
            max_sentence = key 
    if max_value > threshold:
        return max_sentence
    else:
        return None


def filter_metrics(results, ap_thresh=0.5, ar_thresh=0.5):
    """
    Filter categories by AP and AR thresholds and save filtered results to a JSON file.
    """
    filtered = [r for r in results if r["AP"] >= ap_thresh and r["AR"] >= ar_thresh]
    return filtered

def save_evaluation_results(results, filepath = "tests_simplified/raw_data/evaluation_results_model.json"):
    """
    Save a list of results (dicts) to a JSON file with pretty formatting.
    """
    pd.DataFrame(results).to_json(filepath, orient="records", indent=2)

def generate_pdf_table(results, pdf_path):
    """
    Generate pretty PDF table summary.
    """
    if not results:
        print("\n No categories met the AP and AR threshold criteria.")
        return

    df = pd.DataFrame(results).sort_values("AP", ascending=False).reset_index(drop=True)
    df_display = df.copy()
    df_display["AP"] = df_display["AP"].map("{:.3f}".format)
    df_display["AR"] = df_display["AR"].map("{:.3f}".format)

    fig, ax = plt.subplots(figsize=(10, len(df_display) * 0.4 + 1))
    ax.axis('off')
    table = ax.table(cellText=df_display.values,
                     colLabels=df_display.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#cccccc')

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"\n📄 Saved filtered results PDF as '{pdf_path}'")


def save_timing_memory_stats(prompt_start_time, total_image_time, all_image_paths, peak_memories):
    """
    Calculate timing and GPU memory stats for a prompt, then save the summary to stats_summary dictionary.
    """
    stats_summary = {}
    prompt_total_time = time.time() - prompt_start_time
    avg_time_per_image = total_image_time / len(all_image_paths) if all_image_paths else 0
    avg_memory = sum(peak_memories) / len(peak_memories) if peak_memories else 0

    stats_summary[f"question_prompt"] = {
        "total_time_sec": prompt_total_time,
        "average_time_per_image_sec": avg_time_per_image,
        "average_peak_gpu_memory_MB": avg_memory
    }
    return stats_summary

def mean_positive(values):
    valid_values = values[values > -1]
    return float('nan') if valid_values.size == 0 else np.mean(valid_values)

def print_detections(ground_truth_labels, ground_truth_boxes, results):
    """
    Print ground truth labels and boxes, then print detection results.
    """
    print("\nGround truth results:")
    for label, boxes in zip(ground_truth_labels, ground_truth_boxes):
        for  b in boxes:
            print(f"{label}: {b}")

    print("\nFinal detection results:")
    for label, boxes in results.items():
        for box in boxes:
            print(f" {label}: {box}")