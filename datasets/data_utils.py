import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
import json
import pandas as pd
from pathlib import Path
import pydicom
import nibabel as nib
from PIL import Image
import cv2
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import h5py


class MedicalImageLoader:
    """
    Unified loader for various medical image formats
    """

    def __init__(self):
        self.supported_formats = {
            '.dcm': self._load_dicom,
            '.nii': self._load_nifti,
            '.nii.gz': self._load_nifti,
            '.mhd': self._load_mhd,
            '.png': self._load_image,
            '.jpg': self._load_image,
            '.jpeg': self._load_image,
            '.tiff': self._load_image,
            '.npy': self._load_numpy,
            '.npz': self._load_numpy_compressed,
            '.h5': self._load_hdf5,
            '.hdf5': self._load_hdf5
        }

    def load(self, path: Union[str, Path]) -> np.ndarray:
        """Load medical image from path"""
        path = Path(path)

        # Get file extension
        if path.suffix == '.gz' and path.stem.endswith('.nii'):
            ext = '.nii.gz'
        else:
            ext = path.suffix.lower()

        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {ext}")

        return self.supported_formats[ext](path)

    def _load_dicom(self, path: Path) -> np.ndarray:
        """Load DICOM file"""
        ds = pydicom.dcmread(str(path))

        # Apply modality LUT if present
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            array = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        else:
            array = ds.pixel_array

        return array.astype(np.float32)

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load NIfTI file"""
        img = nib.load(str(path))
        array = img.get_fdata()

        # Reorient to RAS if needed
        # This ensures consistent orientation
        return array.astype(np.float32)

    def _load_mhd(self, path: Path) -> np.ndarray:
        """Load MetaImage (mhd/raw) file"""
        img = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(img)
        return array.astype(np.float32)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load standard image formats"""
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

    def _load_numpy(self, path: Path) -> np.ndarray:
        """Load numpy array"""
        return np.load(str(path))

    def _load_numpy_compressed(self, path: Path) -> np.ndarray:
        """Load compressed numpy array"""
        data = np.load(str(path))
        # Assume the main array is stored under 'arr_0' or 'data'
        if 'data' in data:
            return data['data']
        elif 'arr_0' in data:
            return data['arr_0']
        else:
            # Return the first array found
            return list(data.values())[0]

    def _load_hdf5(self, path: Path, dataset_name: str = 'data') -> np.ndarray:
        """Load HDF5 file"""
        with h5py.File(str(path), 'r') as f:
            if dataset_name in f:
                return f[dataset_name][:]
            else:
                # Return the first dataset found
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        return f[key][:]
        raise ValueError(f"No dataset found in {path}")


class DataAugmentation:
    """
    Medical image specific data augmentation
    """

    def __init__(
            self,
            rotation_range: float = 15,
            width_shift_range: float = 0.1,
            height_shift_range: float = 0.1,
            zoom_range: float = 0.1,
            horizontal_flip: bool = True,
            vertical_flip: bool = False,
            brightness_range: Tuple[float, float] = (0.9, 1.1),
            contrast_range: Tuple[float, float] = (0.9, 1.1),
            noise_std: float = 0.01
    ):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentations"""
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = self._rotate(image, angle)
            if mask is not None:
                mask = self._rotate(mask, angle, interpolation=cv2.INTER_NEAREST)

        # Random shifts
        if self.width_shift_range > 0 or self.height_shift_range > 0:
            h, w = image.shape[:2]
            tx = np.random.uniform(-self.width_shift_range, self.width_shift_range) * w
            ty = np.random.uniform(-self.height_shift_range, self.height_shift_range) * h
            image = self._translate(image, tx, ty)
            if mask is not None:
                mask = self._translate(mask, tx, ty, interpolation=cv2.INTER_NEAREST)

        # Random zoom
        if self.zoom_range > 0:
            zoom = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            image = self._zoom(image, zoom)
            if mask is not None:
                mask = self._zoom(mask, zoom, interpolation=cv2.INTER_NEAREST)

        # Random flips
        if self.horizontal_flip and np.random.rand() > 0.5:
            image = np.fliplr(image)
            if mask is not None:
                mask = np.fliplr(mask)

        if self.vertical_flip and np.random.rand() > 0.5:
            image = np.flipud(image)
            if mask is not None:
                mask = np.flipud(mask)

        # Brightness adjustment
        if self.brightness_range != (1.0, 1.0):
            factor = np.random.uniform(*self.brightness_range)
            image = image * factor

        # Contrast adjustment
        if self.contrast_range != (1.0, 1.0):
            factor = np.random.uniform(*self.contrast_range)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Add noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise

        # Clip values
        image = np.clip(image, 0, 1) if image.max() <= 1 else np.clip(image, 0, 255)

        return image, mask

    def _rotate(self, image: np.ndarray, angle: float, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Rotate image"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=interpolation)
        return rotated

    def _translate(self, image: np.ndarray, tx: float, ty: float, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Translate image"""
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = image.shape[:2]
        translated = cv2.warpAffine(image, M, (w, h), flags=interpolation)
        return translated

    def _zoom(self, image: np.ndarray, factor: float, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Zoom image"""
        h, w = image.shape[:2]

        # Calculate crop
        new_h, new_w = int(h / factor), int(w / factor)
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop and resize
        cropped = image[top:top + new_h, left:left + new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=interpolation)

        return zoomed


class DataPreprocessor:
    """
    Preprocessing utilities for medical data
    """

    @staticmethod
    def normalize_intensity(
            image: np.ndarray,
            method: str = 'minmax',
            percentiles: Tuple[float, float] = (1, 99)
    ) -> np.ndarray:
        """
        Normalize image intensity

        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore', 'percentile')
            percentiles: Percentiles for robust normalization

        Returns:
            Normalized image
        """
        if method == 'minmax':
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image - min_val
        elif method == 'zscore':
            mean = image.mean()
            std = image.std()
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean
        elif method == 'percentile':
            p_low, p_high = np.percentile(image, percentiles)
            normalized = np.clip(image, p_low, p_high)
            if p_high > p_low:
                normalized = (normalized - p_low) / (p_high - p_low)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    @staticmethod
    def resize_with_padding(
            image: np.ndarray,
            target_size: Tuple[int, int],
            padding_value: float = 0
    ) -> np.ndarray:
        """
        Resize image with padding to maintain aspect ratio

        Args:
            image: Input image
            target_size: Target size (height, width)
            padding_value: Value for padding

        Returns:
            Resized and padded image
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)

        # Resize
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if len(image.shape) == 3:
            padded = np.pad(
                resized,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=padding_value
            )
        else:
            padded = np.pad(
                resized,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=padding_value
            )

        return padded

    @staticmethod
    def extract_patches(
            image: np.ndarray,
            patch_size: Tuple[int, int],
            stride: Optional[Tuple[int, int]] = None,
            min_overlap: float = 0.0
    ) -> List[np.ndarray]:
        """
        Extract patches from image

        Args:
            image: Input image
            patch_size: Patch size (height, width)
            stride: Stride for patch extraction
            min_overlap: Minimum overlap ratio

        Returns:
            List of patches
        """
        h, w = image.shape[:2]
        patch_h, patch_w = patch_size

        if stride is None:
            stride = patch_size

        stride_h, stride_w = stride

        patches = []
        for y in range(0, h - patch_h + 1, stride_h):
            for x in range(0, w - patch_w + 1, stride_w):
                patch = image[y:y + patch_h, x:x + patch_w]
                patches.append(patch)

        return patches

    @staticmethod
    def reconstruct_from_patches(
            patches: List[np.ndarray],
            image_size: Tuple[int, int],
            patch_size: Tuple[int, int],
            stride: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Reconstruct image from patches

        Args:
            patches: List of patches
            image_size: Original image size
            patch_size: Patch size
            stride: Stride used for extraction

        Returns:
            Reconstructed image
        """
        h, w = image_size
        patch_h, patch_w = patch_size

        if stride is None:
            stride = patch_size

        stride_h, stride_w = stride

        # Initialize output
        if len(patches[0].shape) == 3:
            reconstructed = np.zeros((h, w, patches[0].shape[2]))
            counts = np.zeros((h, w, patches[0].shape[2]))
        else:
            reconstructed = np.zeros((h, w))
            counts = np.zeros((h, w))

        # Add patches
        idx = 0
        for y in range(0, h - patch_h + 1, stride_h):
            for x in range(0, w - patch_w + 1, stride_w):
                reconstructed[y:y + patch_h, x:x + patch_w] += patches[idx]
                counts[y:y + patch_h, x:x + patch_w] += 1
                idx += 1

        # Average overlapping regions
        reconstructed = reconstructed / np.maximum(counts, 1)

        return reconstructed


class DatasetSplitter:
    """
    Utilities for splitting medical datasets
    """

    @staticmethod
    def split_by_patient(
            data: List[Dict],
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15,
            patient_id_key: str = 'patient_id',
            random_state: int = 42
    ) -> Dict[str, List[Dict]]:
        """
        Split data by patient ID to avoid data leakage

        Args:
            data: List of data items
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            patient_id_key: Key for patient ID in data
            random_state: Random seed

        Returns:
            Dictionary with train/val/test splits
        """
        # Group by patient
        patient_data = {}
        for item in data:
            patient_id = item.get(patient_id_key, 'unknown')
            if patient_id not in patient_data:
                patient_data[patient_id] = []
            patient_data[patient_id].append(item)

        # Get patient list
        patients = list(patient_data.keys())
        np.random.seed(random_state)
        np.random.shuffle(patients)

        # Calculate split indices
        n_patients = len(patients)
        n_train = int(n_patients * train_ratio)
        n_val = int(n_patients * val_ratio)

        # Split patients
        train_patients = patients[:n_train]
        val_patients = patients[n_train:n_train + n_val]
        test_patients = patients[n_train + n_val:]

        # Create splits
        splits = {
            'train': [],
            'val': [],
            'test': []
        }

        for patient in train_patients:
            splits['train'].extend(patient_data[patient])
        for patient in val_patients:
            splits['val'].extend(patient_data[patient])
        for patient in test_patients:
            splits['test'].extend(patient_data[patient])

        return splits

    @staticmethod
    def create_cross_validation_splits(
            data: List[Dict],
            n_folds: int = 5,
            patient_id_key: str = 'patient_id',
            random_state: int = 42
    ) -> List[Dict[str, List[Dict]]]:
        """
        Create cross-validation splits

        Args:
            data: List of data items
            n_folds: Number of folds
            patient_id_key: Key for patient ID
            random_state: Random seed

        Returns:
            List of fold splits
        """
        # Group by patient
        patient_data = {}
        for item in data:
            patient_id = item.get(patient_id_key, 'unknown')
            if patient_id not in patient_data:
                patient_data[patient_id] = []
            patient_data[patient_id].append(item)

        # Shuffle patients
        patients = list(patient_data.keys())
        np.random.seed(random_state)
        np.random.shuffle(patients)

        # Create folds
        fold_size = len(patients) // n_folds
        folds = []

        for i in range(n_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_folds - 1 else len(patients)

            val_patients = patients[val_start:val_end]
            train_patients = patients[:val_start] + patients[val_end:]

            fold = {
                'train': [],
                'val': []
            }

            for patient in train_patients:
                fold['train'].extend(patient_data[patient])
            for patient in val_patients:
                fold['val'].extend(patient_data[patient])

            folds.append(fold)

        return folds