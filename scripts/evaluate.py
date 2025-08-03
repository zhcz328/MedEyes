#!/usr/bin/env python3
"""
Evaluation script for MedEyes model
"""

import argparse
import json
import logging
from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from inference.predictor import MedEyesPredictor, PredictionConfig
from utils.metrics import compute_metrics, compute_trajectory_metrics
from datasets.medical_vqa_dataset import MedicalVQADataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MedEyes model')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['vqa-rad', 'slake', 'pathvqa', 'pmc-vqa', 'mmmu'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate')

    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save prediction visualizations')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./eval_outputs',
                        help='Directory to save outputs')

    return parser.parse_args()


def load_dataset(args) -> DataLoader:
    """Load evaluation dataset"""
    dataset = MedicalVQADataset(
        data_root=Path(args.data_root) / args.dataset,
        split=args.split,
        augment=False
    )

    # Limit samples if specified
    if args.num_samples:
        dataset.data = dataset.data[:args.num_samples]

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    return dataloader


def evaluate_model(
        predictor: MedEyesPredictor,
        dataloader: DataLoader,
        args
) -> Dict[str, float]:
    """Evaluate model on dataset"""
    all_results = []
    all_predictions = []
    all_ground_truths = []
    all_metadata = []

    logger.info(f"Evaluating on {len(dataloader)} batches...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        # Get predictions
        for i in range(len(batch['images'])):
            image = batch['images'][i]
            question = batch['questions'][i]
            ground_truth = batch['answers'][i]
            metadata = batch['metadata'][i] if 'metadata' in batch else {}

            # Make prediction
            try:
                result = predictor.predict(
                    image,
                    question,
                    return_visualization=args.save_visualizations
                )

                # Store results
                all_predictions.append(result['answer'])
                all_ground_truths.append(ground_truth)
                all_metadata.append(metadata)

                # Full result for saving
                full_result = {
                    'idx': batch_idx * args.batch_size + i,
                    'question': question,
                    'prediction': result['answer'],
                    'ground_truth': ground_truth,
                    'metadata': metadata,
                    'inference_time': result.get('inference_time', 0),
                    'reasoning_chain': result.get('reasoning_chain', [])
                }

                # Save visualization if requested
                if args.save_visualizations and 'visualization' in result:
                    viz_path = Path(args.output_dir) / 'visualizations' / f"{full_result['idx']}.png"
                    viz_path.parent.mkdir(parents=True, exist_ok=True)

                    import cv2
                    cv2.imwrite(str(viz_path), cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR))
                    full_result['visualization_path'] = str(viz_path)

                all_results.append(full_result)

            except Exception as e:
                logger.error(f"Error processing sample {batch_idx * args.batch_size + i}: {e}")
                all_predictions.append("")
                all_ground_truths.append(ground_truth)
                all_metadata.append(metadata)

    # Compute metrics
    question_types = [m.get('question_type') for m in all_metadata] if all_metadata else None

    metrics = compute_metrics(
        all_predictions,
        all_ground_truths,
        question_types=question_types
    )

    # Add trajectory metrics if available
    trajectories = [r.get('reasoning_chain', []) for r in all_results if 'reasoning_chain' in r]
    if trajectories:
        trajectory_metrics = compute_trajectory_metrics(trajectories)
        metrics.update({f"trajectory_{k}": v for k, v in trajectory_metrics.items()})

    # Save predictions if requested
    if args.save_predictions:
        save_path = Path(args.output_dir) / f"{args.dataset}_{args.split}_predictions.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Saved predictions to {save_path}")

    return metrics, all_results


def analyze_results(results: List[Dict], metrics: Dict[str, float]) -> Dict:
    """Analyze evaluation results"""
    analysis = {
        'overall_metrics': metrics,
        'error_analysis': {},
        'performance_by_type': {},
        'trajectory_analysis': {}
    }

    # Error analysis
    errors = [r for r in results if r['prediction'].lower() != r['ground_truth'].lower()]
    if errors:
        # Common error patterns
        error_types = {}
        for error in errors:
            # Simple categorization based on answer types
            pred_type = 'yes/no' if error['prediction'].lower() in ['yes', 'no'] else 'other'
            gt_type = 'yes/no' if error['ground_truth'].lower() in ['yes', 'no'] else 'other'

            error_pattern = f"{gt_type} -> {pred_type}"
            error_types[error_pattern] = error_types.get(error_pattern, 0) + 1

        analysis['error_analysis']['error_patterns'] = error_types
        analysis['error_analysis']['error_rate'] = len(errors) / len(results)

    # Performance by question type
    type_performance = {}
    for result in results:
        qtype = result['metadata'].get('question_type', 'unknown')
        if qtype not in type_performance:
            type_performance[qtype] = {'correct': 0, 'total': 0}

        type_performance[qtype]['total'] += 1
        if result['prediction'].lower() == result['ground_truth'].lower():
            type_performance[qtype]['correct'] += 1

    for qtype, stats in type_performance.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    analysis['performance_by_type'] = type_performance

    # Trajectory analysis
    chain_lengths = [len(r.get('reasoning_chain', [])) for r in results]
    if chain_lengths:
        analysis['trajectory_analysis'] = {
            'avg_chain_length': np.mean(chain_lengths),
            'std_chain_length': np.std(chain_lengths),
            'min_chain_length': min(chain_lengths),
            'max_chain_length': max(chain_lengths)
        }

    return analysis


def create_evaluation_report(
        metrics: Dict[str, float],
        analysis: Dict,
        args,
        output_path: Path
) -> None:
    """Create evaluation report"""
    report = f"""# MedEyes Evaluation Report

## Configuration
- **Model**: {args.checkpoint}
- **Dataset**: {args.dataset}
- **Split**: {args.split}
- **Number of samples**: {args.num_samples or 'All'}

## Overall Metrics
"""

    # Add metrics
    for metric, value in metrics.items():
        if isinstance(value, float):
            report += f"- **{metric}**: {value:.4f}\n"
        else:
            report += f"- **{metric}**: {value}\n"

    # Add error analysis
    report += "\n## Error Analysis\n"
    if 'error_analysis' in analysis and analysis['error_analysis']:
        report += f"- **Error Rate**: {analysis['error_analysis']['error_rate']:.2%}\n"
        report += "- **Error Patterns**:\n"
        for pattern, count in analysis['error_analysis'].get('error_patterns', {}).items():
            report += f"  - {pattern}: {count}\n"

    # Add performance by type
    report += "\n## Performance by Question Type\n"
    if 'performance_by_type' in analysis:
        for qtype, stats in analysis['performance_by_type'].items():
            report += f"- **{qtype}**: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n"

    # Add trajectory analysis
    report += "\n## Trajectory Analysis\n"
    if 'trajectory_analysis' in analysis and analysis['trajectory_analysis']:
        for key, value in analysis['trajectory_analysis'].items():
            report += f"- **{key}**: {value:.2f}\n"

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Saved evaluation report to {output_path}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    config = PredictionConfig(
        model_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        return_trajectories=True,
        visualization=args.save_visualizations
    )
    predictor = MedEyesPredictor(config)

    # Load dataset
    logger.info(f"Loading {args.dataset} dataset")
    dataloader = load_dataset(args)

    # Evaluate
    logger.info("Starting evaluation...")
    metrics, results = evaluate_model(predictor, dataloader, args)

    # Analyze results
    logger.info("Analyzing results...")
    analysis = analyze_results(results, metrics)

    # Print metrics
    logger.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")

    # Create report
    report_path = output_dir / f"{args.dataset}_{args.split}_report.md"
    create_evaluation_report(metrics, analysis, args, report_path)

    # Save detailed results
    results_path = output_dir / f"{args.dataset}_{args.split}_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'metrics': metrics,
            'analysis': analysis
        }, f, indent=2)

    logger.info(f"Saved detailed results to {results_path}")
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()