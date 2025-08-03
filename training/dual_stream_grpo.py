import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


class DualStreamGRPO:
    """
    Dual-stream Group Relative Policy Optimization
    """

    def __init__(
            self,
            model: nn.Module,
            config: Dict,
            device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.01
        )

        # Initialize replay buffer
        self.replay_buffer = ExperienceReplayBuffer(
            capacity=10000,
            device=device
        )

        # Statistics tracking
        self.stats = defaultdict(list)

    def train_step(
            self,
            batch: Dict,
            iteration: int
    ) -> Dict[str, float]:
        """Single training step with dual-stream optimization"""

        # Generate on-policy rollouts
        on_policy_data = self._generate_on_policy_rollouts(batch)

        # Sample off-policy data from replay buffer
        off_policy_data = self._sample_off_policy_data(
            batch_size=len(batch['images'])
        )

        # Compute advantages separately
        on_policy_advantages = self._compute_advantages(
            on_policy_data,
            source='on_policy'
        )

        off_policy_advantages = self._compute_advantages(
            off_policy_data,
            source='off_policy'
        )

        # Compute policy loss
        loss = self._compute_dual_stream_loss(
            on_policy_data,
            on_policy_advantages,
            off_policy_data,
            off_policy_advantages
        )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['gradient_clip']
        )

        self.optimizer.step()

        # Add successful trajectories to replay buffer
        self._update_replay_buffer(on_policy_data)

        # Log statistics
        stats = {
            'loss': loss.item(),
            'on_policy_reward': np.mean([d['reward'] for d in on_policy_data]),
            'off_policy_reward': np.mean([d['reward'] for d in off_policy_data]) if off_policy_data else 0,
            'replay_buffer_size': len(self.replay_buffer)
        }

        return stats

    def _generate_on_policy_rollouts(self, batch: Dict) -> List[Dict]:
        """Generate on-policy rollouts"""
        rollouts = []
        n_rollouts = self.config['training']['grpo']['n_rollouts']

        with torch.no_grad():
            for i in range(len(batch['images'])):
                image = batch['images'][i].unsqueeze(0)
                question = batch['questions'][i]

                for _ in range(n_rollouts):
                    # Generate trajectory
                    output = self.model(
                        images=image,
                        questions=[question],
                        mode='inference'
                    )

                    # Compute reward
                    reward = self._compute_reward(
                        output['answers'][0],
                        batch['answers'][i],
                        output['reasoning_chains'][0]
                    )

                    rollouts.append({
                        'image': image,
                        'question': question,
                        'answer': output['answers'][0],
                        'ground_truth': batch['answers'][i],
                        'reasoning_chain': output['reasoning_chains'][0],
                        'reward': reward,
                        'log_prob': self._compute_log_prob(output)
                    })

        return rollouts

    def _sample_off_policy_data(self, batch_size: int) -> List[Dict]:
        """Sample off-policy data from replay buffer"""
        n_off_policy = self.config['training']['grpo']['n_off_policy']

        if len(self.replay_buffer) < n_off_policy:
            return []

        # Sample from replay buffer
        samples = self.replay_buffer.sample(
            min(n_off_policy * batch_size, len(self.replay_buffer))
        )

        # Recompute log probabilities with current policy
        off_policy_data = []
        for sample in samples:
            with torch.no_grad():
                log_prob = self._compute_trajectory_log_prob(
                    sample['image'],
                    sample['question'],
                    sample['reasoning_chain']
                )

            off_policy_data.append({
                **sample,
                'current_log_prob': log_prob,
                'old_log_prob': sample['log_prob']
            })

        return off_policy_data

    def _compute_advantages(
            self,
            data: List[Dict],
            source: str
    ) -> torch.Tensor:
        """Compute advantages with source-specific normalization"""
        if not data:
            return torch.tensor([])

        rewards = torch.tensor([d['reward'] for d in data], dtype=torch.float32)

        # Source-specific normalization
        if source == 'on_policy':
            mean = rewards.mean()
            std = rewards.std() + 1e-8
        else:  # off_policy
            # Use running statistics for off-policy
            mean = self.replay_buffer.get_reward_stats()['mean']
            std = self.replay_buffer.get_reward_stats()['std']

        advantages = (rewards - mean) / std

        return advantages

    def _compute_dual_stream_loss(
            self,
            on_policy_data: List[Dict],
            on_policy_advantages: torch.Tensor,
            off_policy_data: List[Dict],
            off_policy_advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute dual-stream GRPO loss"""

        # On-policy loss
        on_policy_loss = 0
        if on_policy_data:
            for i, data in enumerate(on_policy_data):
                ratio = torch.exp(data['log_prob'] - data['log_prob'].detach())

                # Clipped objective
                obj1 = ratio * on_policy_advantages[i]
                obj2 = torch.clamp(
                    ratio,
                    1 - self.config['training']['grpo']['clip_ratio'],
                    1 + self.config['training']['grpo']['clip_ratio']
                ) * on_policy_advantages[i]

                on_policy_loss += -torch.min(obj1, obj2).mean()

        # Off-policy loss
        off_policy_loss = 0
        if off_policy_data:
            for i, data in enumerate(off_policy_data):
                # Importance sampling ratio
                ratio = torch.exp(
                    data['current_log_prob'] - data['old_log_prob']
                )

                # Clipped objective with importance weighting
                obj1 = ratio * off_policy_advantages[i]
                obj2 = torch.clamp(
                    ratio,
                    1 - self.config['training']['grpo']['clip_ratio'],
                    1 + self.config['training']['grpo']['clip_ratio']
                ) * off_policy_advantages[i]

                off_policy_loss += -torch.min(obj1, obj2).mean()

        # Combine losses
        total_loss = on_policy_loss + off_policy_loss

        # Add KL penalty if configured
        kl_coef = self.config['training']['grpo']['kl_coefficient']
        if kl_coef > 0:
            # Compute KL divergence
            kl_div = self._compute_kl_divergence(on_policy_data)
            total_loss += kl_coef * kl_div

        return total_loss

    def _compute_reward(
            self,
            prediction: str,
            ground_truth: str,
            reasoning_chain: List[Dict]
    ) -> float:
        """Compute composite reward"""
        rewards = {}
        weights = self.config['training']['rewards']

        # Accuracy reward
        rewards['accuracy'] = float(
            prediction.lower().strip() == ground_truth.lower().strip()
        )

        # Grammar reward (format correctness)
        rewards['grammar'] = self._compute_grammar_reward(reasoning_chain)

        # Diversity reward (visual exploration)
        rewards['diversity'] = self._compute_diversity_reward(reasoning_chain)

        # Weighted sum
        total_reward = sum(
            rewards[k] * weights[k] for k in rewards
        )

        return total_reward

    def _compute_grammar_reward(self, reasoning_chain: List[Dict]) -> float:
        """Check format correctness of reasoning chain"""
        required_tags = ['reasoning', 'action', 'answer']
        found_tags = set()

        for step in reasoning_chain:
            if step['type'] in required_tags:
                found_tags.add(step['type'])

        # Check if all required tags are present
        if len(found_tags) == len(required_tags):
            return 1.0
        else:
            return len(found_tags) / len(required_tags)

    def _compute_diversity_reward(self, reasoning_chain: List[Dict]) -> float:
        """Compute visual exploration diversity"""
        explored_regions = []

        for step in reasoning_chain:
            if step['type'] == 'tool_call' and step['tool'] == 'gaze':
                coord = step['parameters'].get('coordinate', [])
                if coord:
                    explored_regions.append(coord)

        if not explored_regions:
            return 0.0

        # Compute IoU between regions
        diversity_score = 0.0
        n_pairs = 0

        for i in range(len(explored_regions)):
            for j in range(i + 1, len(explored_regions)):
                iou = self._compute_iou(
                    explored_regions[i],
                    explored_regions[j]
                )
                if iou < self.config['training']['rewards']['iou_threshold']:
                    diversity_score += 1.0
                n_pairs += 1

        if n_pairs > 0:
            diversity_score /= n_pairs

        return diversity_score

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

    def _compute_log_prob(self, output: Dict) -> torch.Tensor:
        """Compute log probability of generated trajectory"""
        # Simplified - in practice would compute actual sequence log probs
        return torch.tensor(0.0, requires_grad=True)

    def _compute_trajectory_log_prob(
            self,
            image: torch.Tensor,
            question: str,
            reasoning_chain: List[Dict]
    ) -> torch.Tensor:
        """Compute log probability of a specific trajectory"""
        # Re-run model with teacher forcing on the trajectory
        # Implementation details...
        return torch.tensor(0.0)

    def _compute_kl_divergence(self, data: List[Dict]) -> torch.Tensor:
        """Compute KL divergence for regularization"""
        # Implementation details...
        return torch.tensor(0.0)

    def _update_replay_buffer(self, on_policy_data: List[Dict]):
        """Add successful trajectories to replay buffer"""
        for data in on_policy_data:
            if data['reward'] > 0.5:  # Only add successful trajectories
                self.replay_buffer.add(data)


class ExperienceReplayBuffer:
    """Experience replay buffer for off-policy training"""

    def __init__(self, capacity: int, device: str = 'cuda'):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
        self.reward_stats = {
            'mean': 0.0,
            'std': 1.0,
            'count': 0
        }

    def add(self, experience: Dict):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

        # Update reward statistics
        self._update_reward_stats(experience['reward'])

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch from buffer"""
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )
        return [self.buffer[i] for i in indices]

    def _update_reward_stats(self, reward: float):
        """Update running reward statistics"""
        n = self.reward_stats['count']
        old_mean = self.reward_stats['mean']

        # Incremental mean update
        new_mean = (old_mean * n + reward) / (n + 1)

        # Incremental variance update
        if n > 0:
            old_var = self.reward_stats['std'] ** 2
            new_var = (n * old_var + (reward - old_mean) * (reward - new_mean)) / (n + 1)
            new_std = np.sqrt(new_var)
        else:
            new_std = 0.0

        self.reward_stats.update({
            'mean': new_mean,
            'std': new_std + 1e-8,
            'count': n + 1
        })

    def get_reward_stats(self) -> Dict[str, float]:
        """Get reward statistics"""
        return self.reward_stats

    def __len__(self):
        return len(self.buffer)