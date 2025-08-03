import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
from .grn import GazeGuidedReasoningNavigator
from .cvs import ConfidenceValueSampler, CVSConfig
from .medplib_integration import MedPLIBWrapper
from .qwen_vl_wrapper import QwenVLWrapper


class MedEyes(nn.Module):
    """
    MedEyes: Dynamic Visual Focus Learning for Medical Diagnosis
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Initialize Qwen2.5-VL backbone
        self.vision_language_model = QwenVLWrapper(
            model_name=config['model']['backbone'],
            device_map="auto"
        )

        # Initialize MedPLIB for segmentation
        self.medplib = MedPLIBWrapper(
            checkpoint_path=config['model']['medplib']['checkpoint'],
            enable_segmentation=config['model']['medplib']['enable_segmentation']
        )

        # Initialize GRN
        self.grn = GazeGuidedReasoningNavigator(
            n_regions=config['grn']['n_regions'],
            confidence_threshold=config['grn']['confidence_threshold'],
            mode_transition_delta=config['grn']['mode_transition_delta'],
            stability_epsilon=config['grn']['stability_epsilon'],
            scanning_prompt=config['grn']['scanning_prompt'],
            drilling_prompt=config['grn']['drilling_prompt']
        )

        # Initialize CVS
        cvs_config = CVSConfig(
            nucleus_p=config['cvs']['nucleus_p'],
            max_trajectory_length=config['cvs']['max_trajectory_length'],
            n_expert_trajectories=config['cvs']['n_expert_trajectories'],
            termination_confidence=config['cvs']['termination_confidence'],
            temperature=config['cvs']['temperature']
        )
        self.cvs = ConfidenceValueSampler(cvs_config)

        # Tool registry
        self.tools = {
            'gaze': self._gaze_tool
        }

    def forward(
            self,
            images: torch.Tensor,
            questions: List[str],
            mode: str = 'inference'
    ) -> Dict:
        """
        Forward pass for MedEyes

        Args:
            images: Batch of images [B, C, H, W]
            questions: List of questions
            mode: 'inference' or 'training'

        Returns:
            Dictionary containing predictions and trajectories
        """
        batch_size = images.shape[0]
        outputs = {
            'answers': [],
            'trajectories': [],
            'reasoning_chains': []
        }

        for i in range(batch_size):
            image = images[i].unsqueeze(0)
            question = questions[i]

            if mode == 'training':
                # Generate expert trajectories for training
                trajectories = self._generate_expert_trajectories(image, question)
                outputs['trajectories'].append(trajectories)

            # Generate reasoning chain
            reasoning_chain = self._generate_reasoning_chain(image, question)
            outputs['reasoning_chains'].append(reasoning_chain)

            # Extract final answer
            answer = self._extract_answer(reasoning_chain)
            outputs['answers'].append(answer)

        return outputs

    def _generate_expert_trajectories(
            self,
            image: torch.Tensor,
            question: str
    ) -> List[Dict]:
        """Generate expert trajectories using CVS"""
        # Extract image features
        image_features = self.vision_language_model.encode_image(image)

        # Sample trajectories
        trajectories = self.cvs.sample_trajectories(
            self.grn,
            image_features,
            self.medplib,
            question
        )

        # Parse trajectories into dialog format
        parsed_trajectories = self.cvs.parse_trajectories(trajectories)

        return parsed_trajectories

    def _generate_reasoning_chain(
            self,
            image: torch.Tensor,
            question: str
    ) -> List[Dict]:
        """Generate reasoning chain for a single image-question pair"""
        reasoning_chain = []

        # Initialize conversation
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]

        # Add system prompt
        system_prompt = self._get_system_prompt()
        messages.insert(0, {"role": "system", "content": system_prompt})

        # Generate reasoning with tool use
        max_turns = self.config['cvs']['max_trajectory_length']

        for turn in range(max_turns):
            # Generate response
            response = self.vision_language_model.generate(
                messages,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )

            # Parse response for tool calls
            tool_calls = self._parse_tool_calls(response)

            if tool_calls:
                # Execute tools and get feedback
                for tool_call in tool_calls:
                    feedback = self._execute_tool(
                        tool_call['name'],
                        tool_call['parameters'],
                        image
                    )

                    reasoning_chain.append({
                        'type': 'tool_call',
                        'tool': tool_call['name'],
                        'parameters': tool_call['parameters'],
                        'feedback': feedback
                    })

                    # Add feedback to messages
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "tool",
                        "content": feedback
                    })
            else:
                # No tool calls, check if answer is provided
                if "<answer>" in response:
                    reasoning_chain.append({
                        'type': 'answer',
                        'content': response
                    })
                    break
                else:
                    reasoning_chain.append({
                        'type': 'reasoning',
                        'content': response
                    })

            # Check termination conditions
            if self._should_terminate(reasoning_chain):
                break

        return reasoning_chain

    def _get_system_prompt(self) -> str:
        """Get system prompt for medical VQA"""
        return """You are an expert medical AI assistant capable of analyzing medical images. 
        When answering questions:
        1. First think about what regions of the image are relevant (<reasoning>)
        2. Use the gaze tool to focus on specific regions (<action>)
        3. Analyze the visual evidence carefully
        4. Provide a clear, medically accurate answer (<answer>)

        Available tools:
        - gaze: Focus on specific image regions {"name": "gaze", "coordinate": [x1, y1, x2, y2]}
        """

    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """Parse tool calls from model response"""
        import re
        tool_calls = []

        # Look for action tags
        action_pattern = r'<action>(.*?)</action>'
        actions = re.findall(action_pattern, response, re.DOTALL)

        for action in actions:
            try:
                import json
                action_dict = json.loads(action)
                tool_calls.append(action_dict)
            except:
                pass

        return tool_calls

    def _execute_tool(
            self,
            tool_name: str,
            parameters: Dict,
            image: torch.Tensor
    ) -> str:
        """Execute a tool and return feedback"""
        if tool_name in self.tools:
            return self.tools[tool_name](parameters, image)
        else:
            return f"Unknown tool: {tool_name}"

    def _gaze_tool(self, parameters: Dict, image: torch.Tensor) -> str:
        """Gaze tool for focusing on image regions"""
        coordinate = parameters.get('coordinate', [])
        if len(coordinate) != 4:
            return "Invalid coordinate format. Expected [x1, y1, x2, y2]"

        # Extract region and analyze
        x1, y1, x2, y2 = coordinate

        # Use MedPLIB to analyze the region
        with torch.no_grad():
            analysis = self.medplib.analyze_region(
                image,
                bbox=[x1, y1, x2, y2],
                return_description=True
            )

        return f"<observation>Region [{x1}, {y1}, {x2}, {y2}]: {analysis['description']}</observation>"

    def _zoom_tool(self, parameters: Dict, image: torch.Tensor) -> str:
        """Zoom tool for high-resolution view"""
        coordinate = parameters.get('coordinate', [])
        if len(coordinate) != 4:
            return "Invalid coordinate format"

        # Crop and resize region
        x1, y1, x2, y2 = coordinate
        # Implementation details...

        return "<observation>Zoomed view shows detailed structures...</observation>"

    def _segment_tool(self, parameters: Dict, image: torch.Tensor) -> str:
        """Segmentation tool using MedPLIB"""
        target = parameters.get('target', 'all')

        with torch.no_grad():
            segments = self.medplib.segment(image, target=target)

        return f"<observation>Found {len(segments)} segments: {', '.join(segments.keys())}</observation>"

    def _should_terminate(self, reasoning_chain: List[Dict]) -> bool:
        """Check if reasoning should terminate"""
        # Terminate if answer is found
        for step in reasoning_chain:
            if step['type'] == 'answer':
                return True

        # Terminate if max length reached
        if len(reasoning_chain) >= self.config['cvs']['max_trajectory_length'] * 2:
            return True

        return False

    def _extract_answer(self, reasoning_chain: List[Dict]) -> str:
        """Extract final answer from reasoning chain"""
        for step in reversed(reasoning_chain):
            if step['type'] == 'answer':
                content = step['content']
                # Extract answer from tags
                import re
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if match:
                    return match.group(1).strip()

        return "Unable to determine answer"