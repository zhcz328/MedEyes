import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVLWrapper(nn.Module):
    """
    Qwen2.5-VL model
    """

    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
            device_map: str = "auto",
            torch_dtype: str = "auto",
            attn_implementation: str = "flash_attention_2"
    ):
        super().__init__()

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Set generation config
        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id
        }

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to features

        Args:
            image: Input image tensor [B, C, H, W]

        Returns:
            Image features
        """
        # Process image through vision encoder
        with torch.no_grad():
            # Convert image tensor to PIL if needed
            if isinstance(image, torch.Tensor):
                # Assuming image is already preprocessed
                pixel_values = image
            else:
                # Process with processor
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )
                pixel_values = inputs["pixel_values"]

            # Get vision features
            vision_outputs = self.model.vision_tower(
                pixel_values=pixel_values
            )

            # Extract features
            features = vision_outputs.last_hidden_state

        return features

    def generate(
            self,
            messages: List[Dict],
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            do_sample: Optional[bool] = None
    ) -> str:
        """
        Generate response given messages

        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample

        Returns:
            Generated text response
        """
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Update generation config
        gen_config = self.generation_config.copy()
        if max_new_tokens is not None:
            gen_config["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["top_p"] = top_p
        if do_sample is not None:
            gen_config["do_sample"] = do_sample

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_config
            )

        # Decode response
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        response = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return response

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass for training

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            pixel_values: Image pixel values
            labels: Target labels for training

        Returns:
            Model outputs including loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

        return outputs

    def prepare_inputs_for_training(
            self,
            messages: List[Dict],
            labels: List[str]
    ) -> Dict:
        """
        Prepare inputs for training

        Args:
            messages: List of message dictionaries
            labels: List of target labels

        Returns:
            Dictionary of model inputs
        """
        # Apply chat template with labels
        texts = []
        for msg_list, label in zip(messages, labels):
            # Add assistant response as label
            msg_list_with_label = msg_list + [{
                "role": "assistant",
                "content": label
            }]

            text = self.processor.apply_chat_template(
                msg_list_with_label,
                tokenize=False
            )
            texts.append(text)

        # Process inputs
        image_inputs = []
        for msg_list in messages:
            imgs, _ = process_vision_info(msg_list)
            image_inputs.extend(imgs)

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Create labels
        # Mask input tokens, only compute loss on response tokens
        labels = inputs["input_ids"].clone()

        # Simple approach: mask everything before "assistant" token
        # In practice, would need more sophisticated masking
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels

        return inputs