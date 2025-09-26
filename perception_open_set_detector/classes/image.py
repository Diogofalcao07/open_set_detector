import os
import numpy as np
from perception_open_set_detector.classes.detection import GroundTruthDetection

class ImageData:

    def __init__(self, image_path, coco_data = None):
        self.image_path = image_path
        self.coco_data = coco_data
        self.image_meta = self._load_image_meta() if self.coco_data is not None else None
        self.annotations = self._load_annotations() if self.coco_data is not None else None
        self.ground_truth_objects = []
        self.detections = [] # Array of Model_Detections()

    def _load_image_meta(self):
        image_name = os.path.basename(self.image_path)
        image_meta = next((img for img in self.coco_data.get('images', []) if img.get('file_name') == image_name), None)
        if image_meta is None:
            raise ValueError(f"No metadata found for image: {image_name}")
        return image_meta
    
    def _load_annotations(self):
        # Filter annotations matching this image's ID
        image_id = self.image_meta["id"]
        return [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]

    def get_annotation_count(self):
        return len(self.annotations)
    
    def get_ground_truth_labels(self):
        if not self.ground_truth_objects:
            return []
        return [gt.label for gt in self.ground_truth_objects]
    
    def add_model_detections(self, detections_list):
        self.detections.extend(detections_list)
    
    def add_ground_truth_detections(self, detections_list):
        self.ground_truth_objects.extend(detections_list)

    def add_ground_truth_detections(self, detections_list):
        self.ground_truth_objects.extend(detections_list)

    def create_ground_truth_detections(self):
        ground_truth_detections = []
        for ann in self.annotations:
            # Convert bbox [x, y, width, height] to [x_min, y_min, x_max, y_max]
            box = np.array([
                ann["bbox"][0],
                ann["bbox"][1],
                ann["bbox"][0] + ann["bbox"][2],
                ann["bbox"][1] + ann["bbox"][3]
            ])

            # Find corresponding category name
            label = next(
                (cat["name"] for cat in self.coco_data["categories"] if cat["id"] == ann["category_id"]),
                str(ann["category_id"])
            )
            ground_truth_detections.append(GroundTruthDetection(label=label, bbox=box))

        self.add_ground_truth_detections(ground_truth_detections)