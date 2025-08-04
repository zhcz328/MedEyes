import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter
import json


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
            reasoning_chain: Complete reasoning chain with reasoning/action/feedback steps
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
        components['feedback'] = self.calculate_feedback_reward(reasoning_chain)  # New feedback reward

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
        """Calculate grammar/format correctness reward for MedEyes structure"""
        if not reasoning_chain:
            return 0.0

        # Check required components
        has_reasoning = any(step['type'] == 'reasoning' for step in reasoning_chain)
        has_action = any(step['type'] == 'action' for step in reasoning_chain)
        has_answer = any(step['type'] == 'answer' for step in reasoning_chain)

        # Check format validity for each step type
        format_scores = []
        for step in reasoning_chain:
            if step['type'] == 'reasoning':
                score = self._check_reasoning_format(step.get('content', ''))
                format_scores.append(score)
            elif step['type'] == 'action':
                score = self._check_action_format(step)
                format_scores.append(score)
            elif step['type'] == 'feedback':
                score = self._check_feedback_format(step)
                format_scores.append(score)

        # Base score from required components
        base_score = sum([has_reasoning, has_action, has_answer]) / 3.0

        # Format correctness
        format_score = np.mean(format_scores) if format_scores else 1.0

        return base_score * format_score

    def calculate_diversity_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate visual exploration diversity reward"""
        explored_regions = []
        action_types = set()

        for step in reasoning_chain:
            if step['type'] == 'action':
                # Parse action content for gaze coordinates
                action_content = step.get('content', '')
                action_types.add(self._extract_action_type(action_content))

                # Extract coordinates from action
                coordinates = self._extract_coordinates_from_action(action_content)
                if coordinates:
                    explored_regions.append(coordinates)

        if not explored_regions:
            return 0.0

        # Calculate spatial diversity
        spatial_diversity = self._calculate_spatial_diversity(explored_regions)

        # Action diversity bonus
        action_diversity = len(action_types) / 2.0  # Normalize by expected number of action types

        return 0.8 * spatial_diversity + 0.2 * min(action_diversity, 1.0)

    def calculate_grounding_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate visual grounding quality reward"""
        grounding_scores = []

        for i, step in enumerate(reasoning_chain):
            if step['type'] == 'action':
                # Check if subsequent feedback and reasoning reference the action
                feedback_score = 0.0
                reasoning_score = 0.0

                # Check for feedback after action
                if i + 1 < len(reasoning_chain) and reasoning_chain[i + 1]['type'] == 'feedback':
                    feedback_score = 1.0  # Feedback present

                    # Check if next reasoning references the visual information
                    if i + 2 < len(reasoning_chain) and reasoning_chain[i + 2]['type'] == 'reasoning':
                        reasoning_content = reasoning_chain[i + 2].get('content', '').lower()
                        visual_terms = ['image', 'region', 'area', 'shows', 'visible', 'observe', 'see', 'examine']
                        if any(term in reasoning_content for term in visual_terms):
                            reasoning_score = 1.0

                combined_score = 0.4 * feedback_score + 0.6 * reasoning_score
                grounding_scores.append(combined_score)

        return np.mean(grounding_scores) if grounding_scores else 0.0

    def calculate_feedback_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate feedback utilization reward"""
        feedback_scores = []

        for i, step in enumerate(reasoning_chain):
            if step['type'] == 'feedback':
                score = 0.0

                # Check if feedback follows an action
                if i > 0 and reasoning_chain[i - 1]['type'] == 'action':
                    score += 0.5  # Proper sequence

                # Check if feedback content is meaningful (not just "...")
                feedback_content = step.get('content', '').strip()
                if feedback_content and feedback_content != '...' and len(feedback_content) > 3:
                    score += 0.3  # Has meaningful content
                else:
                    score += 0.3  # Allow placeholder feedback as shown in examples

                # Check if feedback is followed by reasoning that utilizes it
                if i + 1 < len(reasoning_chain) and reasoning_chain[i + 1]['type'] == 'reasoning':
                    next_reasoning = reasoning_chain[i + 1].get('content', '').lower()
                    utilization_terms = ['from', 'based on', 'shows', 'reveals', 'indicates', 'previous', 'observed']
                    if any(term in next_reasoning for term in utilization_terms):
                        score += 0.2  # Feedback utilization

                feedback_scores.append(min(score, 1.0))

        return np.mean(feedback_scores) if feedback_scores else 0.0

    def calculate_efficiency_reward(self, reasoning_chain: List[Dict]) -> float:
        """Calculate reasoning efficiency reward"""
        if not reasoning_chain:
            return 0.0

        # Count reasoning rounds (reasoning-action-feedback cycles)
        reasoning_rounds = sum(1 for step in reasoning_chain if step['type'] == 'reasoning')
        optimal_rounds = 3  # Based on paper examples (2-3 rounds)

        if reasoning_rounds <= optimal_rounds:
            efficiency = 1.0
        else:
            # Decay for longer chains
            efficiency = optimal_rounds / reasoning_rounds

        # Bonus for structured progression
        structured_bonus = self._calculate_structure_bonus(reasoning_chain)
        efficiency = 0.8 * efficiency + 0.2 * structured_bonus

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

            # Valid transitions for MedEyes structure
            valid_transitions = {
                'reasoning': ['action', 'answer'],
                'action': ['feedback'],
                'feedback': ['reasoning', 'answer'],
                'answer': []
            }

            curr_type = curr_step['type']
            next_type = next_step['type']

            if next_type in valid_transitions.get(curr_type, []):
                coherence_scores.append(1.0)
            else:
                coherence_scores.append(0.3)  # Penalize invalid transitions

        return np.mean(coherence_scores) if coherence_scores else 1.0

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = ' '.join(answer.split())
        return answer

    def _numerical_accuracy(self, pred: str, gt: str) -> float:
        """Calculate accuracy for numerical answers"""
        try:
            pred_nums = re.findall(r'\d+\.?\d*', pred)
            gt_nums = re.findall(r'\d+\.?\d*', gt)

            if not pred_nums or not gt_nums:
                return 0.0

            pred_val = float(pred_nums[0])
            gt_val = float(gt_nums[0])

            if gt_val != 0:
                error = abs(pred_val - gt_val) / abs(gt_val)
                return max(0, 1 - error)
            else:
                return 1.0 if pred_val == gt_val else 0.0

        except:
            return 0.0

    def _multiple_choice_accuracy(self, pred: str, gt: str) -> float:
        """Calculate accuracy for multiple choice questions"""
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

        # Should be wrapped in reasoning tags
        if '<reasoning>' in content and '</reasoning>' in content:
            return 1.0
        else:
            return 0.5  # Content exists but not properly tagged

    def _check_action_format(self, step: Dict) -> float:
        """Check action format validity"""
        content = step.get('content', '')

        # Should contain action tags with JSON
        if '<action>' in content and '</action>' in content:
            # Try to extract and parse JSON
            try:
                action_match = re.search(r'<action>(.*?)</action>', content, re.DOTALL)
                if action_match:
                    json_str = action_match.group(1).strip()
                    parsed = json.loads(json_str)

                    # Check for required fields
                    if 'name' in parsed and parsed['name'] == 'Gaze':
                        if 'coordinate' in parsed and isinstance(parsed['coordinate'], list):
                            if len(parsed['coordinate']) == 4:
                                return 1.0
                return 0.7
            except:
                return 0.3
        return 0.0

    def _check_feedback_format(self, step: Dict) -> float:
        """Check feedback format validity"""
        content = step.get('content', '')

        # Feedback can be simple "..." or more detailed
        if content.strip():
            return 1.0
        return 0.0

    def _extract_action_type(self, action_content: str) -> str:
        """Extract action type from action content"""
        try:
            action_match = re.search(r'<action>(.*?)</action>', action_content, re.DOTALL)
            if action_match:
                json_str = action_match.group(1).strip()
                parsed = json.loads(json_str)
                return parsed.get('name', 'unknown')
        except:
            pass
        return 'unknown'

    def _extract_coordinates_from_action(self, action_content: str) -> Optional[List[float]]:
        """Extract coordinates from action content"""
        try:
            action_match = re.search(r'<action>(.*?)</action>', action_content, re.DOTALL)
            if action_match:
                json_str = action_match.group(1).strip()
                parsed = json.loads(json_str)
                if 'coordinate' in parsed and isinstance(parsed['coordinate'], list):
                    coord = parsed['coordinate']
                    if len(coord) == 4:
                        return [float(x) for x in coord]
        except:
            pass
        return None

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

    def _calculate_structure_bonus(self, reasoning_chain: List[Dict]) -> float:
        """Calculate bonus for proper structure"""
        if not reasoning_chain:
            return 0.0

        # Check if follows reasoning -> action -> feedback pattern
        pattern_score = 0.0
        i = 0
        cycles = 0

        while i < len(reasoning_chain) - 2:
            if (reasoning_chain[i]['type'] == 'reasoning' and
                    reasoning_chain[i + 1]['type'] == 'action' and
                    reasoning_chain[i + 2]['type'] == 'feedback'):
                cycles += 1
                i += 3
            else:
                i += 1

        # Bonus for having complete cycles
        if cycles > 0:
            pattern_score = min(cycles / 3.0, 1.0)  # Up to 3 cycles expected

        return pattern_score

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        try:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[0] + box1[2], box2[0] + box2[2])  # Assuming [x, y, w, h] format
            y2 = min(box1[1] + box1[3], box2[1] + box2[3])

            if x2 < x1 or y2 < y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)
            area1 = box1[2] * box1[3]
            area2 = box2[2] * box2[3]
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0
        except:
            return 0.0


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
            reasoning_chain: List of dicts with 'type' and 'content' keys
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