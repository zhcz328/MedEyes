import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CVSConfig:
    """Configuration for Confidence Value Sampler"""
    nucleus_p: float = 0.9
    max_trajectory_length: int = 4
    n_expert_trajectories: int = 6
    termination_confidence: float = 0.85
    temperature: float = 1.0


class ConfidenceValueSampler(nn.Module):
    """
    Confidence Value Sampler (CVS) for generating expert trajectories
    """

    def __init__(self, config: CVSConfig):
        super().__init__()
        self.config = config
        self.trajectory_buffer = []

    def sample_trajectories(
            self,
            grn: nn.Module,
            image_features: torch.Tensor,
            expert_model: nn.Module,
            query: str
    ) -> List[Dict]:
        """
        Sample multiple expert trajectories using nucleus sampling

        Args:
            grn: Gaze-guided Reasoning Navigator
            image_features: Image features
            expert_model: Expert model for region analysis
            query: Input query

        Returns:
            List of expert trajectories
        """
        trajectories = []

        for i in range(self.config.n_expert_trajectories):
            # Reset GRN state for each trajectory
            grn.reset_state()
            trajectory = self._sample_single_trajectory(
                grn, image_features, expert_model, query
            )
            trajectories.append(trajectory)

        return trajectories

    def _sample_single_trajectory(
            self,
            grn: nn.Module,
            image_features: torch.Tensor,
            expert_model: nn.Module,
            query: str
    ) -> Dict:
        """Sample a single expert trajectory"""
        trajectory = {
            'query': query,
            'steps': [],
            'final_confidence': 0.0,
            'answer': None
        }

        for t in range(self.config.max_trajectory_length):
            # Get action from GRN
            action, state = grn(image_features, expert_model, query)

            # Apply nucleus sampling to action selection
            if action['action_type'] == 'drill' and 'regions' in state.regions:
                action = self._apply_nucleus_sampling(action, state)

            # Add step to trajectory
            step = {
                'step_idx': t,
                'action': action,
                'state': {
                    'mode': state.mode.value,
                    'n_regions': len(state.regions),
                    'avg_confidence': np.mean(list(state.confidences.values())) if state.confidences else 0
                }
            }
            trajectory['steps'].append(step)

            # Check termination condition
            if self._should_terminate(state):
                trajectory['final_confidence'] = max(state.confidences.values()) if state.confidences else 0
                break

        # Generate answer based on final state
        trajectory['answer'] = self._generate_answer(state, expert_model)

        return trajectory

    def _apply_nucleus_sampling(self, action: Dict, state) -> Dict:
        """Apply nucleus sampling to region selection"""
        if not state.regions:
            return action

        # Get confidence scores for all regions
        confidences = []
        for region in state.regions:
            conf = state.confidences.get(region['region_id'], 0.5)
            confidences.append(conf)

        # Convert to probabilities with temperature
        probs = torch.tensor(confidences, dtype=torch.float32)
        probs = F.softmax(probs / self.config.temperature, dim=0)

        # Apply nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)

        # Find nucleus
        nucleus_mask = cumsum_probs <= self.config.nucleus_p
        if not nucleus_mask.any():
            nucleus_mask[0] = True

        # Sample from nucleus
        nucleus_probs = sorted_probs[nucleus_mask]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        nucleus_indices = sorted_indices[nucleus_mask]
        selected_idx = torch.multinomial(nucleus_probs, 1).item()
        selected_region_idx = nucleus_indices[selected_idx].item()

        # Update action with selected region
        action['region_id'] = selected_region_idx
        action['region'] = state.regions[selected_region_idx]

        return action

    def _should_terminate(self, state) -> bool:
        """Check if trajectory should terminate"""
        if not state.confidences:
            return False

        # Terminate if maximum confidence exceeds threshold
        max_confidence = max(state.confidences.values())
        return max_confidence >= self.config.termination_confidence

    def _generate_answer(self, state, expert_model) -> str:
        """Generate answer based on final state"""
        if not state.regions:
            return "No abnormalities detected"

        # Find most confident region
        best_region_id = max(state.confidences.items(), key=lambda x: x[1])[0]
        best_region = state.regions[best_region_id]

        # Generate answer using expert model
        with torch.no_grad():
            answer = expert_model.generate_answer(
                regions=state.regions,
                confidences=state.confidences,
                primary_region=best_region
            )

        return answer

    def parse_trajectories(self, trajectories: List[Dict]) -> List[Dict]:
        """
        Parse raw trajectories into structured dialog sequences

        Args:
            trajectories: List of raw trajectories

        Returns:
            List of parsed dialog sequences
        """
        parsed_trajectories = []

        for traj in trajectories:
            dialog = []

            for step in traj['steps']:
                # Add reasoning step
                reasoning = self._format_reasoning(step)
                dialog.append({
                    'type': 'reasoning',
                    'content': reasoning
                })

                # Add action step
                action = self._format_action(step['action'])
                dialog.append({
                    'type': 'action',
                    'content': action
                })

                # Add feedback (if available)
                if 'feedback' in step:
                    dialog.append({
                        'type': 'feedback',
                        'content': step['feedback']
                    })

            # Add final answer
            dialog.append({
                'type': 'answer',
                'content': traj['answer']
            })

            parsed_trajectories.append({
                'query': traj['query'],
                'dialog': dialog,
                'final_confidence': traj['final_confidence']
            })

        return parsed_trajectories

    def _format_reasoning(self, step: Dict) -> str:
        """Format reasoning step"""
        mode = step['state']['mode']
        n_regions = step['state']['n_regions']
        avg_conf = step['state']['avg_confidence']

        if mode == 'scanning':
            return f"<reasoning>Scanning the image for abnormal regions. Found {n_regions} potential areas of interest.</reasoning>"
        else:
            return f"<reasoning>Drilling into specific region for detailed analysis. Current average confidence: {avg_conf:.2f}</reasoning>"

    def _format_action(self, action: Dict) -> str:
        """Format action step"""
        if action['action_type'] == 'scan':
            coords = []
            for region in action['regions']:
                bbox = region['bbox']
                coords.append(f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            return f"<action>{{\"name\": \"Gaze\", \"coordinates\": {coords}}}</action>"
        else:
            bbox = action['region']['bbox']
            return f"<action>{{\"name\": \"Gaze\", \"coordinate\": [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]}}</action>"