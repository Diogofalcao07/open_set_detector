# === Define Prompts ===
from perception_open_set_detector.classes.model import ChatRexModel
from perception_open_set_detector.classes.dataset import RoboflowDataset
from perception_open_set_detector.classes.prompts import Prompt, SpecificPrompt
from perception_open_set_detector.utils.core_functions import process_and_summarize_prompts

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
label_string = ";".join(custom_labels)

prompts = [
    SpecificPrompt(),  # Prompt 1: dynamic ground-truth labels
    Prompt(" Detect the following object classes in this image: {label_string}"),
    Prompt("Identify all objects visible in this image.", custom_labels),
    Prompt("Examine this image and identify all objects you can see.", custom_labels),
    Prompt("Analyze the scene and describe the objects you detect. Mention anything clearly visible and label them appropriately."),
    Prompt("What do you see in this image?")
]

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

# Define Output Paths
detections_save_dir="tests_simplified/coco_data"
summary_save_path="tests_simplified/raw_data/performance_prompts_summary.json"

process_and_summarize_prompts(prompts, dataset, model, detections_save_dir, summary_save_path)