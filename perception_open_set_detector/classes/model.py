# Imports for ChatRex loading
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from chatrex.upn import UPNWrapper
from PIL import Image
import torch
from perception_open_set_detector.utils.helper_functions import create_detections

class BaseModel():
    """
    Abstract base model class with a standard loader interface.
    """
    def load_models(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def detections(self):
        raise NotImplementedError("Subclasses must implement this method.")   

class ChatRexModel(BaseModel):
    """
    ChatRex model loader that loads processor, ChatRex model, and UPN model.
    """
    def __init__(self, chatrex_model_name="IDEA-Research/ChatRex-7B", upn_checkpoint_path="checkpoints/upn_checkpoints/upn_large.pth", device="cuda"):
        self.chatrex_model_name = chatrex_model_name
        self.upn_checkpoint_path = upn_checkpoint_path
        self.device = device
        self.processor = None
        self.chatrex_model = None
        self.upn_model = None

    def load_processor(self):
        print("Loading ChatRex processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.chatrex_model_name,
            trust_remote_code=True,
            device_map=self.device,
        )

    def load_chatrex_model(self):
        print("Loading ChatRex model...")
        self.chatrex_model = AutoModelForCausalLM.from_pretrained(
            self.chatrex_model_name,
            trust_remote_code=True,
            use_safetensors=True,
        ).to(self.device)

    def load_upn_model(self):
        print("Loading UPN model...")
        self.upn_model = UPNWrapper(self.upn_checkpoint_path)

    def load_models(self):
        self.load_processor()
        self.load_chatrex_model()
        self.load_upn_model()
    
    #### Não sei onde devo colocar esta função

    def detections(self, image, question_prompt, min_score=0.3, nms_value=0.8, max_new_tokens=512):
        """
        Perform object detection using ChatRex and UPN models.
        """

        question_prompt.specific_ground_truth_prompt(image.get_ground_truth_labels())

        # Get UPN predictions
        fine_grained_proposals = self.upn_model.inference(image.image_path, prompt_type="fine_grained_prompt")
        fine_grained_filtered_proposals = self.upn_model.filter(
            fine_grained_proposals, min_score, nms_value )

        # Process inputs for ChatRex
        inputs = self.processor.process(
            image=Image.open(image.image_path).convert("RGB"),
            question=question_prompt.question_prompt,
            bbox=fine_grained_filtered_proposals["original_xyxy_boxes"][0],
        )

        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Configure generation parameters
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=(
                self.processor.tokenizer.pad_token_id
                if self.processor.tokenizer.pad_token_id is not None
                else self.processor.tokenizer.eos_token_id
            ),
        )

        # Perform inference  
        with torch.autocast(self.device, enabled=True, dtype=torch.bfloat16):
            prediction = self.chatrex_model.generate(
                inputs, gen_config=gen_config, tokenizer=self.processor.tokenizer
            )
        question_prompt.prediction = prediction

        # Inference end timing and memory measurement
        stats = question_prompt.prompt_stats
        stats.stop_image_time()  # calls synchronize and computes elapsed internally
        stats.record_peak_memory()
    
        # Get detection boxes
        detection_boxes = fine_grained_filtered_proposals["original_xyxy_boxes"][0]

        detections = create_detections(image, prediction, detection_boxes, question_prompt.custom_labels) 
        image.add_model_detections(detections)
