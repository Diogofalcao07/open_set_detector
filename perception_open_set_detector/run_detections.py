from perception_open_set_detector.classes.model import ChatRexModel
from perception_open_set_detector.classes.dataset import RoboflowDataset
from perception_open_set_detector.classes.prompts import Prompt
from perception_open_set_detector.utils.core_functions import process_images, save_detections_coco, save_detections

# Test close questions with data labels with model
custom_labels = [
    "apple", "bag", "banana", "baseball", "basket", "bin", "bowl", "candle", "cereals",
    "chair", "cheezit", "chocolate_jello", "cleanser", "coffee_grounds", "cola",
    "computer", "cornflakes", "cup", "dice", "dinosaur_toy", "fork", "glass", "gun",
    "iced_tea", "juice_pack", "kitchen-utensils", "knife", "lemon", "milk", "mini_sponge",
    "mustard", "orange", "orange_juice", "peach", "pear", "pen", "phone", "plate",
    "plum", "pringles", "red_wine", "redbull", "remote", "robocup", "rubiks_cube", "shoe",
    "sink", "soccer_ball", "spam", "sponge", "spoon", "strawberry", "strawberry_jello",
    "sugar", "tennis_ball", "tomato_soup", "tropical_juice", "tuna", "waste_bin",
    "water_bottle"
]

question = "Please examine this image and identify all objects."

# Test open question answer with model
question_prompt = Prompt(question, custom_labels)

# Get Dataset
api_key = "6MfL6mP1m3R6aZAnpZ0d"
workspace = "diogofalcao"
#project_name = "robocup2023-l56vz-eunwt"
project_name = "dataset_pic1"
version = 1

###############
dataset = RoboflowDataset(api_key,workspace,project_name,version)
dataset.load()

# Load Model (ChatRex)
model = ChatRexModel()
model.load_models()

all_detections_from_prompt = process_images(dataset, model, question_prompt)

output_path_raw = "perception_open_set_detector/raw_data/model_detections.pkl"
save_detections(all_detections_from_prompt)

output_path_coco = "perception_open_set_detector/coco_data/test_question_detections.json"
save_detections_coco(all_detections_from_prompt) #Use this in evaluator file

