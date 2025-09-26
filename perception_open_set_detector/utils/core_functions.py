import torch, json, pickle
from perception_open_set_detector.classes.image import ImageData
from perception_open_set_detector.utils.helper_functions import mean_positive
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

## Functions to create the detection of 1 or many images

def process_image(image, model, question_prompt):
    stats = question_prompt.prompt_stats
    stats.start_image_time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Get image ground truth
    if image.coco_data is not None:
        image.create_ground_truth_detections()

    # Get image prediction
    model.detections(image, question_prompt)

    return image.detections

def process_images(dataset, model, question_prompt):
    # Data for Prompt Performance
    stats = question_prompt.prompt_stats
    stats.start_prompt_timer()

    pbar = tqdm(dataset.image_paths, desc="Processing images")

    for image_path in pbar:
        image_data = ImageData(image_path, dataset.coco_data)
        image_detections = process_image(image_data, model, question_prompt)
        question_prompt.add_image_detections(image_detections)
    
    stats.get_summary(len(dataset.image_paths))

    return question_prompt.prompt_detections

## Save function files

def save_detections(detections, output_path = "tests_simplified/raw_data/model_detections.pkl"):
    with open(output_path, 'wb') as f:
        pickle.dump(detections, f)
    print(f"\nSaved all predictions to {output_path}")

def save_detections_coco(detections, filename='tests_simplified/coco_data/test_question_detections.json'):
    detections_dicts = [det.to_dict() for det in detections if det.category_id is not None]
    with open(filename, 'w') as f:
        json.dump(detections_dicts, f)



def create_coco_evaluator(coco_gt, coco_dt, all_detections):
    """
    Create and configure COCOeval instance.
    """
    eval = COCOeval(coco_gt, coco_dt, "bbox")
    eval.params.imgIds = list({d["image_id"] for d in all_detections})
    eval.params.catIds = list({d["category_id"] for d in all_detections})
    return eval

def extract_category_metrics(coco_eval, coco_gt):
    """
    Extract AP and AR per category as a list of dicts.
    """
    precisions = coco_eval.eval['precision']
    recalls = coco_eval.eval['recall']
    categories = coco_gt.loadCats(coco_eval.params.catIds)
    id_to_name = {cat['id']: cat['name'] for cat in categories}

    results = []
    for idx, cat_id in enumerate(coco_eval.params.catIds):
        precision = precisions[:, :, idx, 0, 2] 
        ap = mean_positive(precision)

        recall = recalls[:, idx, 0, 2]
        ar = mean_positive(recall)
        print(f"Category '{id_to_name[cat_id]}': AP = {ap:.3f} and AR = {ar:.3f}")
        results.append({"Category": id_to_name[cat_id], "AP": ap, "AR": ar})

    return results

def process_and_summarize_prompts(prompts, dataset, model, detections_save_dir, summary_save_path):
    """
    Process images for each prompt, save detections, and store timing summaries.
    """
    timing_summary = {}

    for index, prompt in enumerate(prompts, start=1):
        all_detections_from_prompt = process_images(dataset, model, prompt)
        output_path_coco = f"{detections_save_dir}/test_prompt_{index}_detector.json"
        save_detections_coco(all_detections_from_prompt, filename=output_path_coco)

        timing_summary[f"prompt_{index}"] = {
            "total_time_sec": prompt.prompt_stats.prompt_total_time,
            "average_time_per_image_sec": prompt.prompt_stats.avg_time_per_image,
            "average_peak_gpu_memory_MB": prompt.prompt_stats.avg_memory
        }

    save_detections(timing_summary, summary_save_path)
