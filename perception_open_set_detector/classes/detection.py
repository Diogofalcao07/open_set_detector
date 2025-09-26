
class Model_Detection:
    def __init__(self, label= None, bbox= None):
        """
        Args:
            label (str): The class label of the detection.
            bbox (list or tuple): Bounding box coordinates [x_min, y_min, x_max, y_max].
        """
        self.label = label
        self.image_id = None
        self.category_id = None
        self.bbox = bbox
        self.score = 1.0
    
    def to_coco(self, image):
        """
        Convert the detection to COCO format.
        """
        categories_id = image.coco_data["categories"]
        category_id = None       
        # Find category id matching the label
        for cat in categories_id:
            if cat["name"].lower() == self.label.lower():
                category_id = cat["id"]
                break
        if category_id is None:
            print(f"Warning: label '{self.label}' not found in COCO categories, skipping")
            return None

        x_min, y_min, x_max, y_max = self.bbox
        width = x_max - x_min
        height = y_max - y_min
        coco_box = [x_min, y_min, width, height]

        self.image_id = int(image.image_meta["id"])
        self.category_id = category_id
        self.bbox = [float(coord) for coord in coco_box]
    
    def to_dict(self):
        return {
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox,
            "score": self.score
            #"label": self.label  # opcional
        }

class GroundTruthDetection:
    def __init__(self, label, bbox):
        """
        Args:
            label (str): The class label of the ground truth detection.
            bbox (list or tuple): Bounding box coordinates [x_min, y_min, x_max, y_max].
        """
        self.label = label
        self.bbox = bbox
