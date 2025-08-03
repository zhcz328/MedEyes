import torch
from torch.utils.data import Dataset
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from torchvision import transforms


class MedicalVQADataset(Dataset):
    """
    Dataset for Medical Visual Question Answering
    """

    def __init__(
            self,
            data_root: Path,
            split: str = 'train',
            image_size: int = 336,
            max_question_length: int = 256,
            augment: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.max_question_length = max_question_length
        self.augment = augment and (split == 'train')

        # Load annotations
        self.annotations = self._load_annotations()

        # Define image transforms
        self.transform = self._create_transform()

    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations"""
        ann_file = self.data_root / f"{self.split}.json"

        if ann_file.exists():
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
        else:
            # Alternative: load from CSV
            csv_file = self.data_root / f"{self.split}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                annotations = df.to_dict('records')
            else:
                raise FileNotFoundError(f"No annotation file found for {self.split}")

        return annotations

    def _create_transform(self):
        """Create image transformation pipeline"""
        if self.augment:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(
                    (self.image_size, self.image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        return transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single item from dataset"""
        ann = self.annotations[idx]

        # Load image
        image_path = self.data_root / 'images' / ann['image']
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        image = self.transform(image)

        # Get question and answer
        question = ann['question']
        answer = ann['answer']

        # Optional: get additional metadata
        metadata = {
            'image_id': ann.get('image_id', idx),
            'question_id': ann.get('question_id', idx),
            'question_type': ann.get('question_type', 'unknown')
        }

        return {
            'image': image,
            'question': question,
            'answer': answer,
            'metadata': metadata
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        images = torch.stack([item['image'] for item in batch])
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        metadata = [item['metadata'] for item in batch]

        return {
            'images': images,
            'questions': questions,
            'answers': answers,
            'metadata': metadata
        }


class MultiDatasetWrapper(Dataset):
    """
    Wrapper for training on multiple medical VQA datasets
    """

    def __init__(
            self,
            dataset_configs: List[Dict],
            sampling_strategy: str = 'balanced'
    ):
        self.datasets = []
        self.dataset_names = []
        self.dataset_sizes = []

        # Load all datasets
        for config in dataset_configs:
            dataset = MedicalVQADataset(**config)
            self.datasets.append(dataset)
            self.dataset_names.append(config['data_root'].name)
            self.dataset_sizes.append(len(dataset))

        self.total_size = sum(self.dataset_sizes)
        self.sampling_strategy = sampling_strategy

        # Create sampling weights
        if sampling_strategy == 'balanced':
            # Equal probability for each dataset
            self.weights = [1.0 / len(self.datasets)] * len(self.datasets)
        elif sampling_strategy == 'proportional':
            # Probability proportional to dataset size
            self.weights = [size / self.total_size for size in self.dataset_sizes]
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict:
        """Get item using weighted sampling"""
        # Sample dataset
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.weights
        )

        # Sample item from selected dataset
        dataset = self.datasets[dataset_idx]
        item_idx = np.random.randint(len(dataset))

        item = dataset[item_idx]
        item['dataset'] = self.dataset_names[dataset_idx]

        return item