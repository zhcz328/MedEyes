import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
from .grn import GazeGuidedReasoningNavigator
from .cvs import ConfidenceValueSampler, CVSConfig
from .medplib_integration import MedPLIBWrapper
from .qwen_vl_wrapper import QwenVLWrapper
import re
import json


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
            'Gaze': self._gaze_tool,  # 匹配论文中的工具名称
            'gaze': self._gaze_tool  # 保持兼容性
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
        """Generate reasoning chain following MedEyes structure"""
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

        # Generate reasoning with tool use following MedEyes pattern
        max_turns = self.config['cvs']['max_trajectory_length']

        for turn in range(max_turns):
            # Generate response
            response = self.vision_language_model.generate(
                messages,
                max_new_tokens=1024,  # Match paper's 1024 token limit
                temperature=0.7,
                do_sample=True
            )

            # Parse different components from response
            components = self._parse_response_components(response)

            # Process each component according to MedEyes structure
            for component in components:
                if component['type'] == 'reasoning':
                    reasoning_chain.append({
                        'type': 'reasoning',
                        'content': component['content']
                    })

                elif component['type'] == 'action':
                    # Execute the action and get feedback
                    tool_name = component['tool_name']
                    parameters = component['parameters']

                    # Add action to chain
                    reasoning_chain.append({
                        'type': 'action',
                        'content': component['raw_content'],
                        'tool_name': tool_name,
                        'parameters': parameters
                    })

                    # Execute tool and get feedback
                    feedback = self._execute_tool(tool_name, parameters, image)

                    # Add feedback to chain
                    reasoning_chain.append({
                        'type': 'feedback',
                        'content': feedback
                    })

                    # Update messages with action and feedback
                    messages.append({"role": "assistant", "content": component['raw_content']})
                    messages.append({"role": "tool", "content": feedback})

                elif component['type'] == 'answer':
                    reasoning_chain.append({
                        'type': 'answer',
                        'content': component['content']
                    })
                    return reasoning_chain  # Terminate when answer is found

            # Check termination conditions
            if self._should_terminate(reasoning_chain):
                break

        return reasoning_chain

    def _parse_response_components(self, response: str) -> List[Dict]:
        """Parse response into reasoning, action, and answer components"""
        components = []

        # Parse reasoning
        reasoning_matches = re.findall(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        for match in reasoning_matches:
            components.append({
                'type': 'reasoning',
                'content': f'<reasoning>{match.strip()}</reasoning>'
            })

        # Parse action
        action_matches = re.findall(r'<action>(.*?)</action>', response, re.DOTALL)
        for match in action_matches:
            try:
                action_dict = json.loads(match.strip())
                components.append({
                    'type': 'action',
                    'raw_content': f'<action>{match.strip()}</action>',
                    'tool_name': action_dict.get('name', ''),
                    'parameters': action_dict
                })
            except json.JSONDecodeError:
                # Handle malformed JSON
                components.append({
                    'type': 'action',
                    'raw_content': f'<action>{match.strip()}</action>',
                    'tool_name': 'unknown',
                    'parameters': {}
                })

        # Parse answer
        answer_matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
        for match in answer_matches:
            components.append({
                'type': 'answer',
                'content': f'<answer>{match.strip()}</answer>'
            })

        return components

    def _get_system_prompt(self) -> str:
        """Get system prompt for medical VQA following MedEyes structure"""
        return """You are an expert medical AI assistant capable of analyzing medical images. 
        When answering questions, follow this structured approach:

        Step 1: Think about what regions of the image are relevant
        <reasoning>Your analysis of what to examine and why</reasoning>

        Step 2: Use the gaze tool to focus on specific regions
        <action>{"name": "Gaze", "coordinate": [x, y, width, height]}</action>

        Step 3: You will receive feedback about the visual region

        Step 4: Repeat steps 1-3 as needed (typically 2-3 rounds)

        Step 5: Provide your final medical answer
        <answer>Your clear, medically accurate conclusion</answer>

        Important notes:
        - Coordinates are in [x, y, width, height] format for a 336x336 image
        - Think systematically about anatomical structures
        - Use medical terminology appropriately
        - Base conclusions on visual evidence

        Example format:
        <reasoning>I need to examine the right lung field for signs of pneumothorax</reasoning>
        <action>{"name": "Gaze", "coordinate": [200, 100, 100, 150]}</action>
        [Feedback will be provided]
        <reasoning>Based on the findings, I can see...</reasoning>
        <answer>Yes, there is evidence of pneumothorax in the right lung</answer>
        """

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
            return f"<feedback>Unknown tool: {tool_name}</feedback>"

    def _gaze_tool(self, parameters: Dict, image: torch.Tensor) -> str:
        """Gaze tool for focusing on image regions"""
        coordinate = parameters.get('coordinate', [])
        if len(coordinate) != 4:
            return "<feedback>Invalid coordinate format. Expected [x, y, width, height]</feedback>"

        # Extract region and analyze
        x, y, w, h = coordinate

        # Validate coordinates
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return "<feedback>Invalid coordinate values</feedback>"

        # Use MedPLIB to analyze the region
        try:
            with torch.no_grad():
                analysis = self.medplib.analyze_region(
                    image,
                    bbox=[x, y, w, h],
                    return_description=True
                )

            # Return structured feedback (can be simplified as "..." in practice)
            return f"<feedback>...</feedback>"  # Simplified as per paper examples

        except Exception as e:
            return f"<feedback>Error analyzing region: {str(e)}</feedback>"

    def _should_terminate(self, reasoning_chain: List[Dict]) -> bool:
        """Check if reasoning should terminate"""
        # Terminate if answer is found
        for step in reasoning_chain:
            if step['type'] == 'answer':
                return True

        # Terminate if max reasoning rounds reached
        reasoning_count = sum(1 for step in reasoning_chain if step['type'] == 'reasoning')
        if reasoning_count >= self.config['cvs']['max_trajectory_length']:
            return True

        return False

    def _extract_answer(self, reasoning_chain: List[Dict]) -> str:
        """Extract final answer from reasoning chain"""
        for step in reversed(reasoning_chain):
            if step['type'] == 'answer':
                content = step['content']
                # Extract answer from tags
                match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if match:
                    return match.group(1).strip()

        return "Unable to determine answer"