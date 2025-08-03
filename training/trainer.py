import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import wandb
from tqdm import tqdm
import json
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for MedEyes training"""
    # Model
    model_name: str = "MedEyes"

    # Training
    num_epochs: int = 3
    batch_size: int = 98
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 500

    # GRPO
    n_rollouts: int = 8
    n_on_policy: int = 8
    n_off_policy: int = 8
    kl_coefficient: float = 0.0
    clip_ratio: float = 0.2

    # Rewards
    reward_weights: Dict[str, float] = None

    # Checkpointing
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 100

    # Output
    output_dir: str = "./outputs"
    resume_from: Optional[str] = None

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'accuracy': 0.7,
                'grammar': 0.2,
                'diversity': 0.1
            }


class MedEyesTrainer:
    """
    Main trainer class for MedEyes
    """

    def __init__(
            self,
            model: nn.Module,
            config: TrainingConfig,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
            device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Move model to device
        self.model = self.model.to(device)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize GRPO trainer
        from .dual_stream_grpo import DualStreamGRPO
        self.grpo_trainer = DualStreamGRPO(
            model=self.model,
            config=config.__dict__,
            device=device
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0

        # Metrics tracking
        self.metrics_history = defaultdict(list)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )

        # Cosine annealing scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
            eta_min=1e-7
        )

        # Combine schedulers
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps]
        )

        return scheduler

    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Training epoch
            train_metrics = self.train_epoch()

            # Log metrics
            self.log_metrics(train_metrics, prefix='train')

            # Validation
            if self.val_loader and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.log_metrics(val_metrics, prefix='val')

                # Save best model
                if val_metrics['accuracy'] > self.best_metric:
                    self.best_metric = val_metrics['accuracy']
                    self.save_checkpoint('best')

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')

        # Final evaluation
        if self.test_loader:
            test_metrics = self.test()
            self.log_metrics(test_metrics, prefix='test')

        # Save final model
        self.save_checkpoint('final')

        logger.info("Training completed!")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        epoch_metrics = defaultdict(list)
        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._prepare_batch(batch)

            # GRPO training step
            step_metrics = self.grpo_trainer.train_step(batch, self.global_step)

            # Update scheduler
            self.scheduler.step()

            # Track metrics
            for k, v in step_metrics.items():
                epoch_metrics[k].append(v)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{step_metrics['loss']:.4f}",
                'reward': f"{step_metrics.get('on_policy_reward', 0):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to wandb
            if self.global_step % self.config.log_interval == 0:
                self.log_step_metrics(step_metrics)

            self.global_step += 1

        # Aggregate epoch metrics
        aggregated_metrics = {}
        for k, v in epoch_metrics.items():
            aggregated_metrics[k] = np.mean(v)

        return aggregated_metrics

    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()

        all_predictions = []
        all_ground_truths = []
        all_metadata = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._prepare_batch(batch)

                # Generate predictions
                outputs = self.model(
                    images=batch['images'],
                    questions=batch['questions'],
                    mode='inference'
                )

                all_predictions.extend(outputs['answers'])
                all_ground_truths.extend(batch['answers'])
                if 'metadata' in batch:
                    all_metadata.extend(batch['metadata'])

        # Compute metrics
        from utils.metrics import compute_metrics
        metrics = compute_metrics(
            all_predictions,
            all_ground_truths,
            question_types=[m.get('question_type') for m in all_metadata] if all_metadata else None
        )

        return metrics

    def test(self) -> Dict[str, float]:
        """Test model"""
        self.model.eval()

        all_predictions = []
        all_ground_truths = []
        all_metadata = []
        all_trajectories = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = self._prepare_batch(batch)

                # Generate predictions with full trajectories
                outputs = self.model(
                    images=batch['images'],
                    questions=batch['questions'],
                    mode='inference'
                )

                all_predictions.extend(outputs['answers'])
                all_ground_truths.extend(batch['answers'])
                all_trajectories.extend(outputs.get('reasoning_chains', []))

                if 'metadata' in batch:
                    all_metadata.extend(batch['metadata'])

        # Compute metrics
        from utils.metrics import compute_metrics, compute_trajectory_metrics

        metrics = compute_metrics(
            all_predictions,
            all_ground_truths,
            question_types=[m.get('question_type') for m in all_metadata] if all_metadata else None
        )

        # Add trajectory metrics
        if all_trajectories:
            trajectory_metrics = compute_trajectory_metrics(all_trajectories)
            metrics.update(trajectory_metrics)

        # Save predictions
        self.save_predictions(all_predictions, all_ground_truths, all_metadata, all_trajectories)

        return metrics

    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare batch for training/inference"""
        if 'images' in batch and isinstance(batch['images'], torch.Tensor):
            batch['images'] = batch['images'].to(self.device)
        return batch

    def log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics"""
        # Add to history
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            self.metrics_history[key].append(v)

        # Log to console
        metric_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"{prefix} metrics: {metric_str}")

        # Log to wandb
        if wandb.run:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics['epoch'] = self.current_epoch
            wandb.log(wandb_metrics)

    def log_step_metrics(self, metrics: Dict[str, float]):
        """Log step-level metrics"""
        if wandb.run:
            wandb_metrics = {f"train/{k}": v for k, v in metrics.items()}
            wandb_metrics['step'] = self.global_step
            wandb.log(wandb_metrics)

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__,
            'metrics_history': dict(self.metrics_history)
        }

        path = self.output_dir / f"checkpoint_{name}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.metrics_history = defaultdict(list, checkpoint.get('metrics_history', {}))

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def save_predictions(
            self,
            predictions: List[str],
            ground_truths: List[str],
            metadata: List[Dict],
            trajectories: List[List[Dict]]
    ):
        """Save test predictions"""
        results = []
        for i in range(len(predictions)):
            result = {
                'id': i,
                'prediction': predictions[i],
                'ground_truth': ground_truths[i],
                'metadata': metadata[i] if i < len(metadata) else {},
                'trajectory': trajectories[i] if i < len(trajectories) else []
            }
            results.append(result)

        output_path = self.output_dir / 'test_predictions.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved predictions to {output_path}")