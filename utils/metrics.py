import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import string


class MedicalVQAMetrics:
    """
    Comprehensive metrics for Medical VQA evaluation
    """

    def __init__(self):
        self.smoothing = SmoothingFunction().method1

    def compute_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute simple accuracy"""
        correct = sum(1 for pred, gt in zip(predictions, ground_truths)
                      if self._normalize_answer(pred) == self._normalize_answer(gt))
        return correct / len(predictions) if predictions else 0.0

    def compute_exact_match(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute exact match score"""
        exact_matches = sum(1 for pred, gt in zip(predictions, ground_truths)
                            if pred.strip().lower() == gt.strip().lower())
        return exact_matches / len(predictions) if predictions else 0.0

    def compute_f1(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute token-level F1 score"""
        f1_scores = []
        for pred, gt in zip(predictions, ground_truths):
            pred_tokens = self._tokenize(pred)
            gt_tokens = self._tokenize(gt)

            if len(gt_tokens) == 0:
                f1_scores.append(0.0 if len(pred_tokens) > 0 else 1.0)
                continue

            common = Counter(pred_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                f1_scores.append(0.0)
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)

        return np.mean(f1_scores) if f1_scores else 0.0

    def compute_bleu(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Compute BLEU scores"""
        bleu_scores = {f'bleu_{i}': [] for i in range(1, 5)}

        for pred, gt in zip(predictions, ground_truths):
            pred_tokens = self._tokenize(pred)
            gt_tokens = self._tokenize(gt)

            for n in range(1, 5):
                weights = [1 / n] * n + [0] * (4 - n)
                score = sentence_bleu(
                    [gt_tokens],
                    pred_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing
                )
                bleu_scores[f'bleu_{n}'].append(score)

        return {k: np.mean(v) if v else 0.0 for k, v in bleu_scores.items()}

    def compute_clinical_accuracy(
            self,
            predictions: List[str],
            ground_truths: List[str],
            question_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute accuracy per clinical question type"""
        if question_types is None:
            return {'overall': self.compute_accuracy(predictions, ground_truths)}

        type_results = {}
        for qtype in set(question_types):
            indices = [i for i, t in enumerate(question_types) if t == qtype]
            if indices:
                preds = [predictions[i] for i in indices]
                gts = [ground_truths[i] for i in indices]
                type_results[qtype] = self.compute_accuracy(preds, gts)

        type_results['overall'] = self.compute_accuracy(predictions, ground_truths)
        return type_results

    def compute_grounding_metrics(
            self,
            pred_bboxes: List[List[float]],
            gt_bboxes: List[List[float]]
    ) -> Dict[str, float]:
        """Compute grounding metrics (IoU, mDice)"""
        ious = []
        dices = []

        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            if pred_bbox and gt_bbox:
                iou = self._compute_iou(pred_bbox, gt_bbox)
                dice = self._compute_dice(pred_bbox, gt_bbox)
                ious.append(iou)
                dices.append(dice)

        return {
            'mean_iou': np.mean(ious) if ious else 0.0,
            'mean_dice': np.mean(dices) if dices else 0.0
        }

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Convert to lowercase
        answer = answer.lower()

        # Remove punctuation
        answer = answer.translate(str.maketrans('', '', string.punctuation))

        # Remove articles
        articles = ['a', 'an', 'the']
        words = answer.split()
        words = [w for w in words if w not in articles]
        answer = ' '.join(words)

        # Handle numbers
        answer = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b',
                        lambda m: str(['zero', 'one', 'two', 'three', 'four', 'five',
                                       'six', 'seven', 'eight', 'nine', 'ten'].index(m.group())),
                        answer)

        return answer.strip()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for F1 computation"""
        # Simple whitespace tokenization
        text = text.lower().strip()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_dice(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Dice coefficient between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        return 2 * intersection / (area1 + area2) if (area1 + area2) > 0 else 0.0


def compute_metrics(
        predictions: List[str],
        ground_truths: List[str],
        question_types: Optional[List[str]] = None,
        pred_bboxes: Optional[List[List[float]]] = None,
        gt_bboxes: Optional[List[List[float]]] = None
) -> Dict[str, float]:
    """
    Compute all metrics for evaluation

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        question_types: Optional list of question types
        pred_bboxes: Optional predicted bounding boxes
        gt_bboxes: Optional ground truth bounding boxes

    Returns:
        Dictionary of computed metrics
    """
    metrics_calculator = MedicalVQAMetrics()

    results = {
        'accuracy': metrics_calculator.compute_accuracy(predictions, ground_truths),
        'exact_match': metrics_calculator.compute_exact_match(predictions, ground_truths),
        'f1': metrics_calculator.compute_f1(predictions, ground_truths)
    }

    # Add BLEU scores
    bleu_scores = metrics_calculator.compute_bleu(predictions, ground_truths)
    results.update(bleu_scores)

    # Add clinical accuracy if question types provided
    if question_types:
        clinical_acc = metrics_calculator.compute_clinical_accuracy(
            predictions, ground_truths, question_types
        )
        results['clinical_accuracy'] = clinical_acc

    # Add grounding metrics if bboxes provided
    if pred_bboxes and gt_bboxes:
        grounding_metrics = metrics_calculator.compute_grounding_metrics(
            pred_bboxes, gt_bboxes
        )
        results.update(grounding_metrics)

    return results


def compute_trajectory_metrics(trajectories: List[Dict]) -> Dict[str, float]:
    """
    Compute metrics for reasoning trajectories

    Args:
        trajectories: List of reasoning trajectories

    Returns:
        Dictionary of trajectory-specific metrics
    """
    metrics = {
        'avg_trajectory_length': 0,
        'avg_tools_used': 0,
        'tool_diversity': 0,
        'reasoning_coherence': 0
    }

    if not trajectories:
        return metrics

    # Compute average trajectory length
    lengths = []
    tools_per_traj = []
    unique_tools = set()

    for traj in trajectories:
        if 'steps' in traj:
            lengths.append(len(traj['steps']))

            # Count tools used
            tools_used = []
            for step in traj['steps']:
                if step.get('action', {}).get('action_type') == 'tool_call':
                    tool = step['action'].get('tool', 'unknown')
                    tools_used.append(tool)
                    unique_tools.add(tool)

            tools_per_traj.append(len(tools_used))

    metrics['avg_trajectory_length'] = np.mean(lengths) if lengths else 0
    metrics['avg_tools_used'] = np.mean(tools_per_traj) if tools_per_traj else 0
    metrics['tool_diversity'] = len(unique_tools)

    # Compute reasoning coherence (simplified)
    coherence_scores = []
    for traj in trajectories:
        if 'final_confidence' in traj:
            coherence_scores.append(traj['final_confidence'])

    metrics['reasoning_coherence'] = np.mean(coherence_scores) if coherence_scores else 0

    return metrics