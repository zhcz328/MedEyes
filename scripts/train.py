import os
import argparse
import yaml
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from tqdm import tqdm

# Import custom modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.medeyes import MedEyes
from datasets.medical_vqa_dataset import MedicalVQADataset
from training.dual_stream_grpo import DualStreamGRPO
from utils.metrics import compute_metrics
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    return rank, world_size, local_rank


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict):
    """Create train/val/test datasets"""
    data_root = Path(config['dataset']['data_root'])
    dataset_name = config['dataset']['name']

    train_dataset = MedicalVQADataset(
        data_root=data_root / dataset_name,
        split='train',
        image_size=config['dataset']['image_size'],
        max_question_length=config['dataset']['max_question_length']
    )

    val_dataset = MedicalVQADataset(
        data_root=data_root / dataset_name,
        split='val',
        image_size=config['dataset']['image_size'],
        max_question_length=config['dataset']['max_question_length']
    )

    return train_dataset, val_dataset


def create_dataloaders(
        train_dataset,
        val_dataset,
        config: dict,
        world_size: int,
        rank: int
):
    """Create distributed dataloaders"""
    batch_size = config['training']['batch_size'] // world_size

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(
        model,
        trainer,
        train_loader,
        epoch,
        config,
        rank
):
    """Train for one epoch"""
    model.train()

    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        pbar = train_loader

    total_stats = {
        'loss': 0,
        'on_policy_reward': 0,
        'off_policy_reward': 0
    }

    for i, batch in enumerate(pbar):
        # Move batch to device
        batch = {
            'images': batch['images'].cuda(),
            'questions': batch['questions'],
            'answers': batch['answers']
        }

        # Training step
        stats = trainer.train_step(batch, epoch * len(train_loader) + i)

        # Accumulate statistics
        for k, v in stats.items():
            if k in total_stats:
                total_stats[k] += v

        # Update progress bar
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{stats['loss']:.4f}",
                'on_reward': f"{stats['on_policy_reward']:.4f}",
                'off_reward': f"{stats['off_policy_reward']:.4f}"
            })

        # Log to wandb
        if rank == 0 and i % config['logging']['log_interval'] == 0:
            wandb.log({
                'train/loss': stats['loss'],
                'train/on_policy_reward': stats['on_policy_reward'],
                'train/off_policy_reward': stats['off_policy_reward'],
                'train/replay_buffer_size': stats['replay_buffer_size'],
                'epoch': epoch,
                'step': epoch * len(train_loader) + i
            })

    # Average statistics
    for k in total_stats:
        total_stats[k] /= len(train_loader)

    return total_stats


def validate(
        model,
        val_loader,
        epoch,
        config,
        rank
):
    """Validate model"""
    model.eval()

    all_predictions = []
    all_ground_truths = []

    if rank == 0:
        pbar = tqdm(val_loader, desc='Validation')
    else:
        pbar = val_loader

    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            batch = {
                'images': batch['images'].cuda(),
                'questions': batch['questions'],
                'answers': batch['answers']
            }

            # Forward pass
            outputs = model(
                images=batch['images'],
                questions=batch['questions'],
                mode='inference'
            )

            all_predictions.extend(outputs['answers'])
            all_ground_truths.extend(batch['answers'])

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_ground_truths)

    # Log to wandb
    if rank == 0:
        wandb.log({
            'val/accuracy': metrics['accuracy'],
            'val/exact_match': metrics['exact_match'],
            'val/f1': metrics['f1'],
            'epoch': epoch
        })

    return metrics


def save_checkpoint(
        model,
        optimizer,
        epoch,
        best_accuracy,
        config,
        is_best=False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'config': config
    }

    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save latest checkpoint
    torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')

    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, output_dir / 'checkpoint_best.pth')

    # Save epoch checkpoint
    if epoch % config['logging']['save_interval'] == 0:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')


def main():
    parser = argparse.ArgumentParser(description='Train MedEyes model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset name')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.output_dir:
        config['logging']['output_dir'] = args.output_dir

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Initialize wandb
    if rank == 0 and config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project_name'],
            config=config,
            name=f"{config['dataset']['name']}_{config['model']['name']}"
        )

    # Create model
    logger.info("Creating model...")
    model = MedEyes(config)
    model = model.cuda()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Create datasets and dataloaders
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        config,
        world_size,
        rank
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = DualStreamGRPO(model, config)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_stats = train_epoch(
            model,
            trainer,
            train_loader,
            epoch,
            config,
            rank
        )

        # Validate
        if epoch % config['logging']['eval_interval'] == 0:
            val_metrics = validate(
                model,
                val_loader,
                epoch,
                config,
                rank
            )

            # Save checkpoint
            if rank == 0:
                is_best = val_metrics['accuracy'] > best_accuracy
                if is_best:
                    best_accuracy = val_metrics['accuracy']

                save_checkpoint(
                    model,
                    trainer.optimizer,
                    epoch,
                    best_accuracy,
                    config,
                    is_best
                )

                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_stats['loss']:.4f}, "
                    f"Val Accuracy={val_metrics['accuracy']:.4f}, "
                    f"Best Accuracy={best_accuracy:.4f}"
                )

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0 and config['logging']['use_wandb']:
        wandb.finish()


if __name__ == '__main__':
    main()