"""
MRI Data Loader
===============
Handles loading of NIfTI (.nii, .nii.gz) and DICOM medical imaging formats.
Implements patient-aware data splitting to prevent data leakage.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import nibabel as nib

try:
    import pydicom
    from pydicom.filereader import dcmread
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
except ImportError:
    SIMPLEITK_AVAILABLE = False


@dataclass
class MRISample:
    """Container for a single MRI sample with metadata."""
    volume: np.ndarray          # 3D array (D, H, W)
    label: int                  # Class label
    patient_id: str            # Patient identifier
    filepath: str              # Original file path
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Voxel spacing
    affine: Optional[np.ndarray] = None  # Affine transformation matrix


def load_nifti(
    filepath: Union[str, Path],
    return_header: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Load a NIfTI file (.nii or .nii.gz).
    
    Args:
        filepath: Path to NIfTI file
        return_header: If True, return header info with volume
        
    Returns:
        3D numpy array of voxel intensities, optionally with header dict
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")
    
    # Load using nibabel
    nii = nib.load(str(filepath))
    volume = nii.get_fdata().astype(np.float32)
    
    if return_header:
        header_info = {
            'affine': nii.affine,
            'spacing': tuple(nii.header.get_zooms()[:3]),
            'shape': volume.shape,
            'dtype': str(volume.dtype),
        }
        return volume, header_info
    
    return volume


def load_dicom_series(
    directory: Union[str, Path],
    return_header: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Load a DICOM series from a directory.
    
    Args:
        directory: Path to directory containing DICOM files
        return_header: If True, return header info with volume
        
    Returns:
        3D numpy array of voxel intensities, optionally with header dict
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"DICOM directory not found: {directory}")
    
    if SIMPLEITK_AVAILABLE:
        # Use SimpleITK for reliable DICOM series loading
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(directory))
        
        if not dicom_files:
            raise ValueError(f"No DICOM series found in: {directory}")
        
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        
        if return_header:
            header_info = {
                'spacing': image.GetSpacing(),
                'origin': image.GetOrigin(),
                'direction': image.GetDirection(),
                'shape': volume.shape,
            }
            return volume, header_info
        return volume
    
    elif PYDICOM_AVAILABLE:
        # Fallback to pydicom
        dicom_files = sorted([
            f for f in directory.iterdir()
            if f.suffix.lower() in ['.dcm', '.dicom', '']
        ])
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in: {directory}")
        
        # Read first file for metadata
        first_slice = dcmread(str(dicom_files[0]))
        
        # Pre-allocate array
        rows = int(first_slice.Rows)
        cols = int(first_slice.Columns)
        slices = []
        
        # Read all slices
        for dcm_file in dicom_files:
            ds = dcmread(str(dcm_file))
            slices.append((float(ds.ImagePositionPatient[2]), ds.pixel_array))
        
        # Sort by position and stack
        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices], axis=0).astype(np.float32)
        
        if return_header:
            header_info = {
                'spacing': (
                    float(first_slice.PixelSpacing[0]),
                    float(first_slice.PixelSpacing[1]),
                    float(first_slice.SliceThickness) if hasattr(first_slice, 'SliceThickness') else 1.0
                ),
                'shape': volume.shape,
            }
            return volume, header_info
        return volume
    
    else:
        raise ImportError("Either SimpleITK or pydicom is required for DICOM loading")


def extract_patient_id(filepath: Union[str, Path]) -> str:
    """
    Extract patient ID from filepath using common naming conventions.
    Falls back to directory-based hashing if no pattern matches.
    
    Args:
        filepath: Path to MRI file
        
    Returns:
        Patient identifier string
    """
    filepath = Path(filepath)
    name = filepath.stem.replace('.nii', '')  # Handle .nii.gz
    
    # Common patterns: sub-001, patient_001, P001, etc.
    patterns = [
        r'(sub-\d+)',
        r'(patient[_-]?\d+)',
        r'(P\d+)',
        r'(ADNI_\d+_S_\d+)',  # ADNI dataset pattern
        r'(\d{3,})',  # Just numbers (3+ digits)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: use parent directory as patient ID
    return filepath.parent.name


class MRIDataset:
    """
    Dataset class for MRI volumes with patient-aware splitting.
    
    Ensures that all scans from the same patient stay in the same split
    to prevent data leakage.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        class_mapping: Optional[Dict[str, int]] = None,
        file_pattern: str = "*.nii.gz",
        cache_volumes: bool = False
    ):
        """
        Initialize MRI dataset.
        
        Args:
            data_dir: Root directory containing class subdirectories
            class_mapping: Optional dict mapping class names to integers
            file_pattern: Glob pattern for finding files
            cache_volumes: If True, cache loaded volumes in memory
        """
        self.data_dir = Path(data_dir)
        self.file_pattern = file_pattern
        self.cache_volumes = cache_volumes
        self._cache: Dict[str, np.ndarray] = {}
        
        # Discover classes if not provided
        if class_mapping is None:
            class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
            class_mapping = {d.name: i for i, d in enumerate(sorted(class_dirs))}
        
        self.class_mapping = class_mapping
        self.num_classes = len(class_mapping)
        
        # Discover all samples
        self.samples: List[Dict] = []
        self.patient_to_indices: Dict[str, List[int]] = {}
        
        self._discover_samples()
    
    def _discover_samples(self):
        """Scan data directory and catalog all samples."""
        for class_name, class_idx in self.class_mapping.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            # Find all matching files
            for filepath in class_dir.glob(self.file_pattern):
                patient_id = extract_patient_id(filepath)
                sample_idx = len(self.samples)
                
                self.samples.append({
                    'filepath': str(filepath),
                    'label': class_idx,
                    'patient_id': patient_id,
                    'class_name': class_name
                })
                
                if patient_id not in self.patient_to_indices:
                    self.patient_to_indices[patient_id] = []
                self.patient_to_indices[patient_id].append(sample_idx)
        
        if not self.samples:
            raise ValueError(f"No samples found in {self.data_dir} with pattern {self.file_pattern}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> MRISample:
        """Load and return a single sample."""
        sample_info = self.samples[idx]
        filepath = sample_info['filepath']
        
        # Check cache
        if self.cache_volumes and filepath in self._cache:
            volume = self._cache[filepath]
        else:
            volume, header = load_nifti(filepath, return_header=True)
            if self.cache_volumes:
                self._cache[filepath] = volume
        
        return MRISample(
            volume=volume,
            label=sample_info['label'],
            patient_id=sample_info['patient_id'],
            filepath=filepath,
            spacing=header.get('spacing', (1.0, 1.0, 1.0)) if 'header' in dir() else (1.0, 1.0, 1.0),
            affine=header.get('affine') if 'header' in dir() else None
        )
    
    def get_patient_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset by patients to prevent data leakage.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        np.random.seed(seed)
        patients = list(self.patient_to_indices.keys())
        np.random.shuffle(patients)
        
        n_patients = len(patients)
        n_train = int(n_patients * train_ratio)
        n_val = int(n_patients * val_ratio)
        
        train_patients = patients[:n_train]
        val_patients = patients[n_train:n_train + n_val]
        test_patients = patients[n_train + n_val:]
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for patient in train_patients:
            train_indices.extend(self.patient_to_indices[patient])
        for patient in val_patients:
            val_indices.extend(self.patient_to_indices[patient])
        for patient in test_patients:
            test_indices.extend(self.patient_to_indices[patient])
        
        return train_indices, val_indices, test_indices
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of samples per class."""
        dist = {}
        for class_name in self.class_mapping:
            dist[class_name] = sum(1 for s in self.samples if s['class_name'] == class_name)
        return dist


class BatchGenerator:
    """
    Memory-efficient batch generator for MRI volumes.
    Loads data incrementally to handle large datasets.
    """
    
    def __init__(
        self,
        dataset: MRIDataset,
        indices: List[int],
        batch_size: int,
        shuffle: bool = True,
        preprocessing_fn: Optional[callable] = None,
        augmentation_fn: Optional[callable] = None
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocessing_fn = preprocessing_fn
        self.augmentation_fn = augmentation_fn
    
    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            
            volumes = []
            labels = []
            
            for idx in batch_indices:
                sample = self.dataset[idx]
                volume = sample.volume
                
                # Apply preprocessing
                if self.preprocessing_fn:
                    volume = self.preprocessing_fn(volume)
                
                # Apply augmentation (training only)
                if self.augmentation_fn:
                    volume = self.augmentation_fn(volume)
                
                # Add channel dimension: (D, H, W) -> (1, D, H, W)
                volume = volume[np.newaxis, ...]
                
                volumes.append(volume)
                labels.append(sample.label)
            
            # Stack into batch: (B, C, D, H, W)
            batch_volumes = np.stack(volumes, axis=0)
            batch_labels = np.array(labels, dtype=np.int64)
            
            yield batch_volumes, batch_labels


def create_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    preprocessing_fn: Optional[callable] = None,
    augmentation_fn: Optional[callable] = None,
    seed: int = 42
) -> Tuple[BatchGenerator, BatchGenerator, BatchGenerator]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root directory containing class subdirectories
        batch_size: Batch size
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        preprocessing_fn: Preprocessing function to apply
        augmentation_fn: Augmentation function for training
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = MRIDataset(data_dir)
    train_idx, val_idx, test_idx = dataset.get_patient_split(
        train_ratio, val_ratio, 1 - train_ratio - val_ratio, seed
    )
    
    train_loader = BatchGenerator(
        dataset, train_idx, batch_size, 
        shuffle=True,
        preprocessing_fn=preprocessing_fn,
        augmentation_fn=augmentation_fn
    )
    
    val_loader = BatchGenerator(
        dataset, val_idx, batch_size,
        shuffle=False,
        preprocessing_fn=preprocessing_fn,
        augmentation_fn=None  # No augmentation for validation
    )
    
    test_loader = BatchGenerator(
        dataset, test_idx, batch_size,
        shuffle=False,
        preprocessing_fn=preprocessing_fn,
        augmentation_fn=None
    )
    
    return train_loader, val_loader, test_loader
