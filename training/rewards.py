import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter


class RewardCalculator:
    """
    Calculate various reward components for MedEyes training
    """

    def __init__(self, config: Dict):
        self.config = config
        self.weights = config.get('weights', {
            'accuracy': 0.7,
            'grammar': 0.2,
            'diversity': 0.1
        })

    def calculate_composite_reward(
            self,
            prediction: str,
            ground_truth: str,
            reasoning_chain: List[Dict],
            metadata: Optional[Dict] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite reward with all components

        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            reasoning_chain: Complete reasoning chain
            metadata: Optional metadata

        Returns:
            Total reward and component breakdown
        """
        components = {}

        # Accuracy reward
        components['accuracy'] = self.calculate_accuracy_reward(
            prediction, ground_truth, metadata
        )

        # Grammar reward
        components['grammar'] = self.calculate_grammar_reward(reasoning_chain)

        # Diversity reward
        components['diversity'] = self.calculate_diversity_reward(reasoning_chain)

        # Additional rewards
        components['grounding'] = self.calculate_grounding_reward(reasoning_chain)
        components['efficiency'] = self.calculate_efficiency_reward(reasoning_chain)
        components['coherence'] = self.calculate_coherence_reward(reasoning_chain)

        # Weighted sum
        total_reward = sum(
            components[k] * self.weights.get(k, 0) for k in components
        )

        return total_reward, components

    def calculate_accuracy_reward(
            self,
            prediction: str,
            ground_truth: str,
            metadata: Optional[Dict] = None
    ) -> float:
        """Calculate accuracy-based reward"""
        # Exact match
        if self._normalize_answer(prediction) == self._normalize_answer(ground_truth):
            return 1.0

        # Partial credit for specific question types
        if metadata and metadata.get('question_type') == 'numerical':
            return self._numerical_accuracy(prediction, ground_truth)
        elif metadata and metadata.get('question_type') == 'multiple_choice':
            return self._multiple_choice_accuracy(prediction, ground_truth)
        else:
            # Token overlap for open-ended questions
            return self._token_overlap_score(prediction, ground_truth)

    def calculate_grammar_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate grammar/format correctness reward"""
        if not reasoning_chain:
            return 0.0

        # Check required components
        has_reasoning = any(step['type'] == 'reasoning' for step in reasoning_chain)
        has_action = any(step['type'] == 'tool_call' for step in reasoning_chain)
        has_answer = any(step['type'] == 'answer' for step in reasoning_chain)

        # Check format validity
        format_scores = []
        for step in reasoning_chain:
            if step['type'] == 'reasoning':
                # Check reasoning format
                score = self._check_reasoning_format(step.get('content', ''))
                format_scores.append(score)
            elif step['type'] == 'tool_call':
                # Check action format
                score = self._check_action_format(step)
                format_scores.append(score)

        # Base score from required components
        base_score = sum([has_reasoning, has_action, has_answer]) / 3.0

        # Format correctness
        format_score = np.mean(format_scores) if format_scores else 1.0

        return base_score * format_score

    def calculate_diversity_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate visual exploration diversity reward"""
        explored_regions = []
        tools_used = set()

        for step in reasoning_chain:
            if step['type'] == 'tool_call':
                tool = step.get('tool', '')
                tools_used.add(tool)

                if tool == 'gaze' and 'parameters' in step:
                    coord = step['parameters'].get('coordinate', [])
                    if len(coord) == 4:
                        explored_regions.append(coord)

        if not explored_regions:
            return 0.0

        # Calculate spatial diversity
        spatial_diversity = self._calculate_spatial_diversity(explored_regions)

        # Tool diversity bonus
        tool_diversity = len(tools_used) / 3.0  # Normalize by expected number of tools

        return 0.7 * spatial_diversity + 0.3 * min(tool_diversity, 1.0)

    def calculate_grounding_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate visual grounding quality reward"""
        grounding_scores = []

        for i, step in enumerate(reasoning_chain):
            if step['type'] == 'tool_call' and step.get('tool') == 'gaze':
                # Check if subsequent reasoning references the region
                if i + 1 < len(reasoning_chain):
                    next_step = reasoning_chain[i + 1]
                    if next_step['type'] == 'reasoning':
                        # Check for spatial references
                        content = next_step.get('content', '').lower()
                        spatial_terms = ['region', 'area', 'location', 'at', 'in', 'shows', 'visible']
                        has_reference = any(term in content for term in spatial_terms)
                        grounding_scores.append(1.0 if has_reference else 0.5)

        return np.mean(grounding_scores) if grounding_scores else 0.0

    def calculate_efficiency_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate reasoning efficiency reward"""
        if not reasoning_chain:
            return 0.0

        # Penalize overly long chains
        length = len(reasoning_chain)
        optimal_length = 4  # Configurable

        if length <= optimal_length:
            efficiency = 1.0
        else:
            # Decay for longer chains
            efficiency = optimal_length / length

        # Bonus for reaching answer quickly
        answer_position = None
        for i, step in enumerate(reasoning_chain):
            if step['type'] == 'answer':
                answer_position = i
                break

        if answer_position is not None:
            early_answer_bonus = 1.0 - (answer_position / length)
            efficiency = 0.7 * efficiency + 0.3 * early_answer_bonus

        return efficiency

    def calculate_coherence_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate reasoning coherence reward"""
        if len(reasoning_chain) < 2:
            return 1.0

        coherence_scores = []

        # Check transitions between steps
        for i in range(len(reasoning_chain) - 1):
            curr_step = reasoning_chain[i]
            next_step = reasoning_chain[i + 1]

            # Valid transitions
            valid_transitions = {
                'reasoning': ['tool_call', 'answer', 'reasoning'],
                'tool_call': ['reasoning', 'answer'],
                'answer': []
            }

            curr_type = curr_step['type']
            next_type = next_step['type']

            if next_type in valid_transitions.get(curr_type, []):
                coherence_scores.append(1.0)
            else:
                coherence_scores.append(0.5)

        return np.mean(coherence_scores) if coherence_scores else 1.0

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Convert to lowercase
        answer = answer.lower().strip()

        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)

        # Remove extra whitespace
        answer = ' '.join(answer.split())

        return answer

    def _numerical_accuracy(self, pred: str, gt: str) -> float:
        """Calculate accuracy for numerical answers"""
        try:
            # Extract numbers
            pred_nums = re.findall(r'\d+\.?\d*', pred)
            gt_nums = re.findall(r'\d+\.?\d*', gt)

            if not pred_nums or not gt_nums:
                return 0.0

            pred_val = float(pred_nums[0])
            gt_val = float(gt_nums[0])

            # Relative error
            if gt_val != 0:
                error = abs(pred_val - gt_val) / abs(gt_val)
                return max(0, 1 - error)
            else:
                return 1.0 if pred_val == gt_val else 0.0

        except:
            return 0.0

    def _multiple_choice_accuracy(self, pred: str, gt: str) -> float:
        """Calculate accuracy for multiple choice questions"""
        # Extract choice letters
        pred_choices = re.findall(r'\b[A-E]\b', pred.upper())
        gt_choices = re.findall(r'\b[A-E]\b', gt.upper())

        if pred_choices and gt_choices:
            return 1.0 if pred_choices[0] == gt_choices[0] else 0.0
        return 0.0

    def _token_overlap_score(self, pred: str, gt: str) -> float:
        """Calculate token overlap F1 score"""
        pred_tokens = self._normalize_answer(pred).split()
        gt_tokens = self._normalize_answer(gt).split()

        if not gt_tokens:
            return 0.0 if pred_tokens else 1.0

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens) if pred_tokens else 0
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    def _check_reasoning_format(self, content: str) -> float:
        """Check reasoning step format"""
        if not content:
            return 0.0

        # Should be wrapped in tags
        if content.startswith('<reasoning>') and content.endswith('</reasoning>'):
            return 1.0
        elif '<reasoning>' in content or '</reasoning>' in content:
            return 0.5
        else:
            return 0.0

    def _check_action_format(self, step: Dict) -> float:
        """Check action format validity"""
        if 'tool' not in step or 'parameters' not in step:
            return 0.0

        tool = step['tool']
        params = step['parameters']

        # Check required parameters for each tool
        if tool == 'gaze':
            if 'coordinate' in params and isinstance(params['coordinate'], list):
                coord = params['coordinate']
                if len(coord) == 4 and all(isinstance(x, (int, float)) for x in coord):
                    return 1.0

        return 0.5

    def _calculate_spatial_diversity(self, regions: List[List[float]]) -> float:
        """Calculate spatial diversity of explored regions"""
        if len(regions) <= 1:
            return 0.0

        # Calculate pairwise IoU
        ious = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                iou = self._compute_iou(regions[i], regions[j])
                ious.append(iou)

        # Diversity is inverse of average IoU
        avg_iou = np.mean(ious) if ious else 0
        diversity = 1.0 - avg_iou

        return diversity

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class CompositeReward:
    """
    Composite reward function with multiple objectives
    """

    def __init__(self, config: Dict):
        self.calculator = RewardCalculator(config)
        self.history = []

    def __call__(
            self,
            prediction: str,
            ground_truth: str,
            reasoning_chain: List[Dict],
            metadata: Optional[Dict] = None
    ) -> float:
        """
        Calculate total reward

        Args:
            prediction: Model prediction
            ground_truth: Ground truth
            reasoning_chain: Reasoning steps
            metadata: Optional metadata

        Returns:
            Total reward value
        """
        reward, components = self.calculator.calculate_composite_reward(
            prediction, ground_truth, reasoning_chain, metadata
        )

        # Store history for analysis
        self.history.append({
            'total': reward,
            'components': components,
            'metadata': metadata
        })

        return reward

    def get_statistics(self) -> Dict[str, float]:
        """Get reward statistics"""
        if not self.history:
            return {}

        stats = {
            'mean_total': np.mean([h['total'] for h in self.history]),
            'std_total': np.std([h['total'] for h in self.history])
        }

        # Component statistics
        component_names = self.history[0]['components'].keys()
        for comp in component_names:
            values = [h['components'][comp] for h in self.history]
            stats[f'mean_{comp}'] = np.mean(values)
            stats[f'std_{comp}'] = np.std(values)

        return stats

    def reset_history(self):
        """Reset reward history"""
        self.history = []