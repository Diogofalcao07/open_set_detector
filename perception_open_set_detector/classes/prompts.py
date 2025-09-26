from perception_open_set_detector.classes.prompt_stats import PromptStats

class Prompt:
    def __init__(self, question_prompt: str, custom_labels=None):
        self.question_prompt = question_prompt + " Answer the question with object indexes."
        self.custom_labels = custom_labels
        self.prediction = None
        self.prompt_detections = []  # Store mapped prediction results here (per image)
        self.prompt_stats = PromptStats()
    
    def add_image_detections(self, detections_list):
        self.prompt_detections.extend(detections_list)
    
    def specific_ground_truth_prompt(self, ground_truth_labels):
        pass


class SpecificPrompt(Prompt):
    def __init__(self, custom_labels=None):
        # Set initial question_prompt empty or placeholder
        super().__init__(question_prompt="", custom_labels=custom_labels)
    
    def specific_ground_truth_prompt(self, ground_truth_labels):
        labels_str = "; ".join(ground_truth_labels)
        self.question_prompt = f"Please detect the following object classes in this image: {labels_str}. Answer with object indexes for each class."
        return self.question_prompt
