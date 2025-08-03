from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import torch
from pathlib import Path


class BaseMedialDataset(Dataset, ABC):
    """
    Abstract base class for medical datasets
    """

    def __init__(
            self,
            data_root: Path,
            split: str = 'train',
            transform: Optional[Any] = None,
            target_transform: Optional[Any] = None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Validate split
        valid_splits = ['train', 'val', 'test']
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")

        # Load data
        self.data = self._load_data()

    @abstractmethod
    def _load_data(self) -> List[Dict]:
        """Load dataset data"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """Get a single item"""
        pass

    def get_metadata(self) -> Dict:
        """Get dataset metadata"""
        return {
            'name': self.__class__.__name__,
            'split': self.split,
            'size': len(self),
            'data_root': str(self.data_root)
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for classification datasets"""
        distribution = {}
        for item in self.data:
            label = item.get('label', item.get('answer', 'unknown'))
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling"""
        class_dist = self.get_class_distribution()
        total_samples = len(self)
        weights = []

        for item in self.data:
            label = item.get('label', item.get('answer', 'unknown'))
            class_count = class_dist[label]
            weight = total_samples / (len(class_dist) * class_count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float)


class MedicalClassificationDataset(BaseMedialDataset):
    """
    Base class for medical classification datasets
    """

    def __init__(
            self,
            data_root: Path,
            split: str = 'train',
            transform: Optional[Any] = None,
            target_transform: Optional[Any] = None,
            num_classes: Optional[int] = None
    ):
        self.num_classes = num_classes
        super().__init__(data_root, split, transform, target_transform)

        # Validate number of classes
        if self.num_classes is None:
            self.num_classes = len(set(item['label'] for item in self.data))

    def get_label_mapping(self) -> Dict[str, int]:
        """Get label to index mapping"""
        unique_labels = sorted(set(item['label'] for item in self.data))
        return {label: idx for idx, label in enumerate(unique_labels)}


class MedicalSegmentationDataset(BaseMedialDataset):
    """
    Base class for medical segmentation datasets
    """

    def __init__(
            self,
            data_root: Path,
            split: str = 'train',
            transform: Optional[Any] = None,
            target_transform: Optional[Any] = None,
            mask_suffix: str = '_mask'
    ):
        self.mask_suffix = mask_suffix
        super().__init__(data_root, split, transform, target_transform)

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        """Load segmentation mask"""
        # Implementation depends on mask format
        pass

    def get_num_classes(self) -> int:
        """Get number of segmentation classes"""
        # Sample a few masks to determine number of classes
        pass


class MedicalDetectionDataset(BaseMedialDataset):
    """
    Base class for medical detection datasets
    """

    def __init__(
            self,
            data_root: Path,
            split: str = 'train',
            transform: Optional[Any] = None,
            target_transform: Optional[Any] = None,
            min_bbox_area: float = 0.0
    ):
        self.min_bbox_area = min_bbox_area
        super().__init__(data_root, split, transform, target_transform)

    def _filter_small_bboxes(self, bboxes: List[List[float]]) -> List[List[float]]:
        """Filter out small bounding boxes"""
        filtered = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area >= self.min_bbox_area:
                filtered.append(bbox)
        return filtered

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for detection"""
        # Handle variable number of bboxes per image
        pass