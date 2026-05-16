import argparse
from pathlib import Path
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run and visualize object detection on a single image using ChatRex"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="perception_open_set_detector/images/test_single_image_detection.jpeg",
        help="Path to the input image (default: perception_open_set_detector/images/test_single_image_detection.jpeg)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/single_image_detection.png",
        help="Path to save the output visualization (default: results/figures/single_image_detection.png)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Examine this image and identify all objects you can see.",
        help="Question/prompt for detection (default: 'Examine this image and identify all objects you can see.')"
    )
    args = parser.parse_args()

    # Validate image path
    if not Path(args.image).exists():
        raise FileNotFoundError(
            f"Image file not found: {args.image}\n"
            f"Please provide a valid image path using --image argument."
        )

    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Inputs ###

    # Create Prompt
    custom_labels = ["candle", "glass", "water_bottle", "pen"]
    question = args.question
    question_prompt = Prompt(question, custom_labels)

    # Load Image
    image_path = args.image
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
    vis_image.save(args.output)
    print(f"Prediction saved to: {args.output}")