import json
from pycocotools.coco import COCO
from perception_open_set_detector.utils.core_functions import create_coco_evaluator, extract_category_metrics
from perception_open_set_detector.utils.helper_functions import filter_metrics, save_evaluation_results, generate_pdf_table

"""
    Evaluates two coco annotations datasets (ground truth and test)
    Generates a pdf summary of the object detections, with a specific threshold for AP and AR
"""

# Get the files
ground_truth_path = "perception_open_set_detector/coco_data/merged_annotations.coco.json"
detections_path = "perception_open_set_detector/coco_data/test_question_detections.json"

with open(detections_path, "r") as f:
    all_detections = json.load(f)

# Prepare the data for COCOeval()
coco_gt = COCO(ground_truth_path)
coco_dt = coco_gt.loadRes(detections_path)

# Procede with the evaluation
coco_eval = create_coco_evaluator(coco_gt, coco_dt, all_detections)

print("\n Overall Metrics:")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

results = extract_category_metrics(coco_eval, coco_gt)
filtered_results = filter_metrics(results)

# Save results 
output_path_data="perception_open_set_detector/raw_data/evaluation_results_model.json"
save_evaluation_results(filtered_results, output_path_data)

output_path_pdf = "perception_open_set_detector/pdfs/filtered_evaluation_results_table_new.pdf"
generate_pdf_table(filtered_results, output_path_pdf)