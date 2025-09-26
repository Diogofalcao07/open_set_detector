import os, json
from roboflow import Roboflow
from perception_open_set_detector.utils.helper_functions import get_all_files_recursive, load_and_merge_coco_annotations_all_subdirs

class DatasetLoader:
    def __init__(self):
        self.dataset = None
        self.coco_data = None
        self.annotation_path = None
        self.image_paths = []

    def load(self):
        raise NotImplementedError("Subclasses must implement this method.")


class RoboflowDataset(DatasetLoader):
    def __init__(self, api_key, workspace, project_name, version=1, data_format="coco"):
        super().__init__()
        self.api_key = api_key
        self.workspace = workspace
        self.project_name = project_name
        self.version = version
        self.data_format = data_format

    def load(self):
        if not all([self.api_key, self.workspace, self.project_name]):
            raise ValueError(
                "api_key, workspace, and project_name must be set to load Roboflow dataset."
            )

        rf = Roboflow(self.api_key)
        project = rf.workspace(self.workspace).project(self.project_name)
        dataset = project.version(self.version).download(self.data_format)
        # annotation_path = os.path.join(dataset.location, "merged_annotations.coco.json")
        annotation_path = "perception_open_set_detector/coco_data/merged_annotations.coco.json"

        coco_data = load_and_merge_coco_annotations_all_subdirs(dataset.location)
        image_paths = get_all_files_recursive(dataset.location)

        with open(annotation_path, "w") as f:
            json.dump(coco_data, f)
        
        print("Dataset downloaded to:", dataset.location)

        self.dataset = dataset
        self.coco_data = coco_data
        self.annotation_path = annotation_path
        self.image_paths = image_paths
