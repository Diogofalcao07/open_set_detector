from PIL import Image
from chatrex.tools.visualize import visualize_chatrex_output
from perception_open_set_detector.classes.model import ChatRexModel
from perception_open_set_detector.classes.prompts import Prompt, SpecificPrompt
from perception_open_set_detector.classes.image import ImageData
from perception_open_set_detector.utils.core_functions import process_image

"""
Run and visualize object detection on a single image using the ChatRex pipeline.
"""

if __name__ == "__main__":
    ### Inputs ###

    # Create Prompt
    custom_labels = ["candle", "glass", "water_bottle", "pen"]
    question = "Examine this image and identify all objects you can see."
    question_prompt = Prompt(question, custom_labels)

    # Load Image
    image_path = "/home/dfalcao/ChatRex/Dataset_PIC1/test/DSC02222_JPG.rf.95a98ae97c4a64f7b1b200eefbf2906b.jpg"
    image = ImageData(image_path)

    # Load Model (ChatRex)
    model = ChatRexModel()
    model.load_models()
    
    # Perform detection
    detections = process_image(image, model, question_prompt)

    # Get detection_boxes (required for visualize_chatrex_output() )
    detection_boxes = [det.bbox for det in detections]

    # Visualize the prediction (this function is from chatrex)
    vis_image = visualize_chatrex_output(
        Image.open(image_path),
        detection_boxes,
        question_prompt.prediction,
        font_size=15,
        draw_width=5,
    )

    # Save prediction image
    vis_image.save("perception_open_set_detector/images/test_single_image_detection.jpeg")
    print(f"prediction is saved at tests_simplified/images/test_single_image_detection.jpeg")