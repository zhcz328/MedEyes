import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import json
import time
from dataclasses import dataclass


@dataclass
class PredictionConfig:
    """Configuration for inference"""
    model_path: str
    device: str = 'cuda'
    batch_size: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    use_cache: bool = True
    return_trajectories: bool = True
    visualization: bool = False


class MedEyesPredictor:
    """
    Predictor class for MedEyes inference
    """

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Cache for repeated queries
        self.cache = {} if config.use_cache else None

    def _load_model(self) -> nn.Module:
        """Load trained MedEyes model"""
        from models.medeyes import MedEyes

        # Load checkpoint
        checkpoint = torch.load(self.config.model_path, map_location=self.device)

        # Extract config
        model_config = checkpoint.get('config', {})

        # Create model
        model = MedEyes(model_config)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        return model

    def predict(
            self,
            image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
            question: str,
            return_visualization: bool = False
    ) -> Dict:
        """
        Make prediction for single image-question pair

        Args:
            image: Input image (path, PIL Image, numpy array, or tensor)
            question: Question about the image
            return_visualization: Whether to return visualization

        Returns:
            Dictionary containing prediction results
        """
        # Check cache
        cache_key = self._get_cache_key(image, question)
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]

        # Prepare input
        image_tensor = self._prepare_image(image)

        # Run inference
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(
                images=image_tensor.unsqueeze(0),
                questions=[question],
                mode='inference'
            )

        inference_time = time.time() - start_time

        # Extract results
        result = {
            'answer': outputs['answers'][0],
            'question': question,
            'inference_time': inference_time
        }

        # Add trajectory if requested
        if self.config.return_trajectories and 'reasoning_chains' in outputs:
            result['reasoning_chain'] = outputs['reasoning_chains'][0]
            result['trajectory_summary'] = self._summarize_trajectory(outputs['reasoning_chains'][0])

        # Add visualization if requested
        if return_visualization or self.config.visualization:
            from .visualization import create_prediction_visualization
            result['visualization'] = create_prediction_visualization(
                image_tensor.cpu().numpy(),
                result.get('reasoning_chain', []),
                result['answer']
            )

        # Cache result
        if self.cache is not None:
            self.cache[cache_key] = result

        return result

    def predict_batch(
            self,
            images: List[Union[str, Path, Image.Image, np.ndarray]],
            questions: List[str]
    ) -> List[Dict]:
        """
        Make predictions for batch of image-question pairs

        Args:
            images: List of input images
            questions: List of questions

        Returns:
            List of prediction results
        """
        if len(images) != len(questions):
            raise ValueError("Number of images and questions must match")

        results = []

        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch_images = images[i:i + self.config.batch_size]
            batch_questions = questions[i:i + self.config.batch_size]

            # Prepare batch
            image_tensors = torch.stack([
                self._prepare_image(img) for img in batch_images
            ])

            # Run inference
            with torch.no_grad():
                outputs = self.model(
                    images=image_tensors,
                    questions=batch_questions,
                    mode='inference'
                )

            # Extract results
            for j in range(len(batch_images)):
                result = {
                    'answer': outputs['answers'][j],
                    'question': batch_questions[j]
                }

                if self.config.return_trajectories and 'reasoning_chains' in outputs:
                    result['reasoning_chain'] = outputs['reasoning_chains'][j]

                results.append(result)

        return results

    def analyze_image(
            self,
            image: Union[str, Path, Image.Image, np.ndarray],
            analysis_type: str = 'comprehensive'
    ) -> Dict:
        """
        Perform comprehensive analysis of medical image

        Args:
            image: Input image
            analysis_type: Type of analysis ('comprehensive', 'abnormalities', 'diagnosis')

        Returns:
            Analysis results
        """
        # Define analysis prompts
        analysis_prompts = {
            'comprehensive': [
                "What abnormalities are visible in this medical image?",
                "Describe the key findings in this image.",
                "What is the most likely diagnosis based on this image?",
                "Are there any urgent findings that require immediate attention?"
            ],
            'abnormalities': [
                "Identify all abnormal regions in this image.",
                "What pathological features are present?",
                "Describe the location and characteristics of any lesions."
            ],
            'diagnosis': [
                "What is the most likely diagnosis?",
                "What is the differential diagnosis?",
                "What additional imaging or tests would you recommend?"
            ]
        }

        prompts = analysis_prompts.get(analysis_type, analysis_prompts['comprehensive'])

        # Run analysis
        analysis_results = {}

        for prompt in prompts:
            result = self.predict(image, prompt)
            analysis_results[prompt] = {
                'answer': result['answer'],
                'confidence': self._extract_confidence(result),
                'key_regions': self._extract_key_regions(result.get('reasoning_chain', []))
            }

        # Synthesize findings
        synthesis = self._synthesize_findings(analysis_results)

        return {
            'analysis_type': analysis_type,
            'detailed_findings': analysis_results,
            'synthesis': synthesis,
            'recommendations': self._generate_recommendations(synthesis)
        }

    def interactive_exploration(
            self,
            image: Union[str, Path, Image.Image, np.ndarray],
            initial_question: str,
            max_turns: int = 5
    ) -> List[Dict]:
        """
        Interactive exploration with follow-up questions

        Args:
            image: Input image
            initial_question: Starting question
            max_turns: Maximum number of interaction turns

        Returns:
            List of interaction turns
        """
        conversation = []
        current_question = initial_question
        context = []

        for turn in range(max_turns):
            # Get prediction
            result = self.predict(image, current_question)

            # Add to conversation
            conversation.append({
                'turn': turn + 1,
                'question': current_question,
                'answer': result['answer'],
                'reasoning_chain': result.get('reasoning_chain', [])
            })

            # Update context
            context.append(f"Q: {current_question}")
            context.append(f"A: {result['answer']}")

            # Generate follow-up question
            follow_up = self._generate_follow_up_question(
                result['answer'],
                result.get('reasoning_chain', []),
                context
            )

            if follow_up is None:
                break

            current_question = follow_up

        return conversation

    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Prepare image for model input"""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(image, np.ndarray):
            # Normalize if needed
            if image.max() > 1:
                image = image / 255.0

            # Convert to tensor
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=2)
            image = torch.from_numpy(image).float()

            # Rearrange dimensions if needed
            if image.shape[2] == 3:
                image = image.permute(2, 0, 1)

        return image.to(self.device)

    def _get_cache_key(self, image: Union[str, Path, Image.Image, np.ndarray], question: str) -> str:
        """Generate cache key"""
        if isinstance(image, (str, Path)):
            image_key = str(image)
        else:
            # Use hash for non-path inputs
            image_key = str(hash(str(image)))

        return f"{image_key}_{question}"

    def _summarize_trajectory(self, reasoning_chain: List[Dict]) -> Dict:
        """Summarize reasoning trajectory"""
        summary = {
            'num_steps': len(reasoning_chain),
            'tools_used': set(),
            'regions_explored': [],
            'key_findings': []
        }

        for step in reasoning_chain:
            if step['type'] == 'tool_call':
                summary['tools_used'].add(step.get('tool', 'unknown'))

                if step.get('tool') == 'gaze' and 'parameters' in step:
                    coord = step['parameters'].get('coordinate', [])
                    if coord:
                        summary['regions_explored'].append(coord)

            elif step['type'] == 'reasoning':
                # Extract key phrases
                content = step.get('content', '')
                if any(term in content.lower() for term in ['abnormal', 'lesion', 'finding']):
                    summary['key_findings'].append(content)

        summary['tools_used'] = list(summary['tools_used'])

        return summary

    def _extract_confidence(self, result: Dict) -> float:
        """Extract confidence score from result"""
        # Simple heuristic - could be improved with actual confidence estimation
        if 'reasoning_chain' in result:
            # More steps might indicate lower confidence
            num_steps = len(result['reasoning_chain'])
            confidence = max(0.5, 1.0 - (num_steps - 3) * 0.1)
        else:
            confidence = 0.8

        return confidence

    def _extract_key_regions(self, reasoning_chain: List[Dict]) -> List[Dict]:
        """Extract key regions from reasoning chain"""
        regions = []

        for i, step in enumerate(reasoning_chain):
            if step['type'] == 'tool_call' and step.get('tool') == 'gaze':
                coord = step.get('parameters', {}).get('coordinate', [])
                if len(coord) == 4:
                    # Look for description in next step
                    description = "Region of interest"
                    if i + 1 < len(reasoning_chain):
                        next_step = reasoning_chain[i + 1]
                        if next_step['type'] == 'reasoning':
                            description = next_step.get('content', description)[:100]

                    regions.append({
                        'bbox': coord,
                        'description': description,
                        'step': i
                    })

        return regions

    def _synthesize_findings(self, analysis_results: Dict) -> Dict:
        """Synthesize findings from multiple analyses"""
        # Extract all findings
        all_findings = []
        abnormalities = []
        diagnoses = []

        for question, result in analysis_results.items():
            answer = result['answer'].lower()

            if 'abnormal' in question.lower():
                abnormalities.append(answer)
            elif 'diagnosis' in question.lower():
                diagnoses.append(answer)

            all_findings.append(answer)

        # Identify consensus
        synthesis = {
            'primary_findings': self._extract_common_themes(abnormalities),
            'diagnostic_impression': self._extract_common_themes(diagnoses),
            'confidence_level': np.mean([r['confidence'] for r in analysis_results.values()])
        }

        return synthesis

    def _extract_common_themes(self, texts: List[str]) -> List[str]:
        """Extract common themes from multiple texts"""
        if not texts:
            return []

        # Simple approach - find common words/phrases
        # In practice, would use more sophisticated NLP
        common_words = set(texts[0].split())
        for text in texts[1:]:
            common_words &= set(text.split())

        return list(common_words)[:5]

    def _generate_recommendations(self, synthesis: Dict) -> List[str]:
        """Generate clinical recommendations based on findings"""
        recommendations = []

        # Based on confidence
        if synthesis['confidence_level'] < 0.7:
            recommendations.append("Consider additional imaging or second opinion due to uncertainty")

        # Based on findings
        if synthesis['primary_findings']:
            recommendations.append("Follow up on identified abnormalities")

        # Generic recommendations
        recommendations.append("Correlate with clinical history and symptoms")

        return recommendations

    def _generate_follow_up_question(
            self,
            answer: str,
            reasoning_chain: List[Dict],
            context: List[str]
    ) -> Optional[str]:
        """Generate follow-up question based on current answer"""
        # Simple heuristic-based approach
        answer_lower = answer.lower()

        # Check for uncertainty
        if any(term in answer_lower for term in ['possibly', 'might be', 'unclear']):
            return "Can you provide more details about the uncertain findings?"

        # Check for abnormalities mentioned
        if 'abnormal' in answer_lower or 'lesion' in answer_lower:
            return "What are the characteristics of the abnormality that support your assessment?"

        # Check if diagnosis was provided
        if any(term in answer_lower for term in ['diagnosis', 'consistent with', 'suggestive of']):
            return "What other conditions should be considered in the differential diagnosis?"

        # Check conversation length
        if len(context) > 6:
            return None  # End conversation

        return "Are there any other significant findings in the image?"