import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum


class ExplorationMode(Enum):
    SCANNING = "scanning"
    DRILLING = "drilling"


@dataclass
class GRNState:
    """State for Gaze-guided Reasoning Navigator"""
    regions: List[Dict[str, float]]  # List of regions with bbox coordinates
    confidences: Dict[int, float]  # Confidence scores for each region
    mode: ExplorationMode  # Current exploration mode
    history: List[Dict]  # Exploration history


class GazeGuidedReasoningNavigator(nn.Module):
    """
    Gaze-guided Reasoning Navigator (GRN) for dual-mode exploration
    """

    def __init__(
            self,
            n_regions: int = 5,
            confidence_threshold: float = 0.85,
            mode_transition_delta: float = 0.15,
            stability_epsilon: float = 1e-6,
            scanning_prompt: str = None,
            drilling_prompt: str = None
    ):
        super().__init__()
        self.n_regions = n_regions
        self.confidence_threshold = confidence_threshold
        self.mode_transition_delta = mode_transition_delta
        self.stability_epsilon = stability_epsilon
        self.scanning_prompt = scanning_prompt or "Please locate all abnormal regions in the image <SEG>"
        self.drilling_prompt = drilling_prompt or "Please analyze the abnormality in region <region>{}</region> <SEG>"

        # Initialize state
        self.reset_state()

    def reset_state(self):
        """Reset GRN state"""
        self.state = GRNState(
            regions=[],
            confidences={},
            mode=ExplorationMode.SCANNING,
            history=[]
        )

    def forward(
            self,
            image_features: torch.Tensor,
            expert_model: nn.Module,
            current_query: str
    ) -> Tuple[Dict, GRNState]:
        """
        Execute one step of exploration

        Args:
            image_features: Image features from vision encoder
            expert_model: Expert model (e.g., MedPLIB) for region analysis
            current_query: Current query or prompt

        Returns:
            action: Dictionary containing action type and parameters
            updated_state: Updated GRN state
        """
        if self.state.mode == ExplorationMode.SCANNING:
            action = self._scanning_mode(image_features, expert_model)
        else:
            action = self._drilling_mode(image_features, expert_model, current_query)

        # Update state based on action results
        self._update_state(action)

        # Check for mode transition
        self._check_mode_transition()

        return action, self.state

    def _scanning_mode(
            self,
            image_features: torch.Tensor,
            expert_model: nn.Module
    ) -> Dict:
        """Execute scanning mode exploration"""
        # Generate prompt for expert model
        prompt = self.scanning_prompt

        # Query expert model for abnormal regions
        with torch.no_grad():
            results = expert_model.detect_regions(
                image_features,
                prompt,
                max_regions=self.n_regions
            )

        # Extract regions and initial confidences
        regions = []
        for i, region in enumerate(results['regions']):
            regions.append({
                'bbox': region['bbox'],  # [x1, y1, x2, y2]
                'confidence': region['confidence'],
                'region_id': i
            })
            self.state.confidences[i] = region['confidence']

        self.state.regions = regions

        return {
            'action_type': 'scan',
            'regions': regions,
            'prompt': prompt
        }

    def _drilling_mode(
            self,
            image_features: torch.Tensor,
            expert_model: nn.Module,
            current_query: str
    ) -> Dict:
        """Execute drilling mode exploration"""
        # Select region with highest uncertainty or lowest confidence
        region_id = self._select_drilling_region()
        region = self.state.regions[region_id]

        # Generate drilling prompt
        prompt = self.drilling_prompt.format(
            f"{region['bbox'][0]},{region['bbox'][1]},{region['bbox'][2]},{region['bbox'][3]}"
        )

        # Perform detailed analysis
        with torch.no_grad():
            results = expert_model.analyze_region(
                image_features,
                region['bbox'],
                prompt
            )

        # Update confidence for this region
        old_confidence = self.state.confidences[region_id]
        new_confidence = results['confidence']
        self.state.confidences[region_id] = new_confidence

        return {
            'action_type': 'drill',
            'region_id': region_id,
            'region': region,
            'old_confidence': old_confidence,
            'new_confidence': new_confidence,
            'analysis': results['analysis'],
            'prompt': prompt
        }

    def _select_drilling_region(self) -> int:
        """Select region for drilling based on confidence scores"""
        # Select region with lowest confidence that hasn't been fully explored
        confidences = list(self.state.confidences.items())
        confidences.sort(key=lambda x: x[1])

        for region_id, conf in confidences:
            if conf < self.confidence_threshold:
                return region_id

        # If all regions have high confidence, select randomly
        return np.random.choice(list(self.state.confidences.keys()))

    def _update_state(self, action: Dict):
        """Update state based on action results"""
        self.state.history.append({
            'step': len(self.state.history),
            'mode': self.state.mode.value,
            'action': action
        })

    def _check_mode_transition(self):
        """Check if mode transition is needed"""
        if self.state.mode == ExplorationMode.SCANNING:
            # Transition to drilling if regions are identified
            if len(self.state.regions) > 0:
                self.state.mode = ExplorationMode.DRILLING
        else:
            # Check confidence evolution for mode transition
            if len(self.state.history) >= 2:
                last_action = self.state.history[-1]['action']
                if last_action['action_type'] == 'drill':
                    delta_c = self._compute_confidence_delta(last_action)
                    if delta_c < self.mode_transition_delta:
                        # Switch back to scanning for broader exploration
                        self.state.mode = ExplorationMode.SCANNING

    def _compute_confidence_delta(self, action: Dict) -> float:
        """Compute confidence evolution"""
        old_conf = action.get('old_confidence', 0)
        new_conf = action.get('new_confidence', 0)
        return abs(new_conf - old_conf) / (old_conf + self.stability_epsilon)

    def get_trajectory(self) -> List[Dict]:
        """Get complete exploration trajectory"""
        return self.state.history