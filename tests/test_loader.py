"""
Test suite for the data loading module.
Run with: python -m pytest tests/test_loader.py -v
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import (
    load_nifti,
    extract_patient_id,
    MRIDataset,
    BatchGenerator
)
from data.preprocessing import (
    normalize_intensity,
    resample_volume,
    PreprocessingPipeline
)
from data.augmentation import (
    random_flip_3d,
    random_rotation_3d,
    AugmentationPipeline
)


class TestNiftiLoader:
    """Tests for NIfTI loading functionality."""
    
    def test_extract_patient_id_standard(self):
        """Test patient ID extraction from standard naming."""
        assert extract_patient_id("sub-001_T1w.nii.gz") == "sub-001"
        assert extract_patient_id("patient_42_session1.nii") == "patient_42"
        
    def test_extract_patient_id_fallback(self):
        """Test patient ID extraction falls back to parent dir."""
        path = Path("/data/patient_001/scan.nii.gz")
        result = extract_patient_id(path)
        assert result == "patient_001"


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_normalize_zscore(self):
        """Test z-score normalization."""
        volume = np.random.rand(32, 32, 32).astype(np.float32) * 100 + 50
        result = normalize_intensity(volume, method="zscore")
        
        # Mean should be ~0, std should be ~1
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1
    
    def test_normalize_minmax(self):
        """Test min-max normalization."""
        volume = np.random.rand(32, 32, 32).astype(np.float32) * 100 + 50
        result = normalize_intensity(volume, method="minmax", clip_percentile=None)
        
        # Values should be in [0, 1]
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_resample_volume(self):
        """Test volume resampling."""
        volume = np.random.rand(64, 64, 64).astype(np.float32)
        target_shape = (32, 32, 32)
        
        result = resample_volume(volume, target_shape)
        
        assert result.shape == target_shape
        assert result.dtype == np.float32
    
    def test_preprocessing_pipeline(self):
        """Test composable preprocessing pipeline."""
        volume = np.random.rand(64, 64, 64).astype(np.float32) * 1000
        
        pipeline = PreprocessingPipeline()
        pipeline.add_normalize("zscore")
        pipeline.add_resample((32, 32, 32))
        
        result = pipeline(volume)
        
        assert result.shape == (32, 32, 32)
        assert abs(result.mean()) < 0.5  # Approximately normalized


class TestAugmentation:
    """Tests for augmentation functions."""
    
    def test_random_flip_shape(self):
        """Test flip maintains shape."""
        volume = np.random.rand(32, 32, 32).astype(np.float32)
        result = random_flip_3d(volume, prob=1.0)  # Force flip
        
        assert result.shape == volume.shape
    
    def test_random_rotation_shape(self):
        """Test rotation maintains shape."""
        volume = np.random.rand(32, 32, 32).astype(np.float32)
        result = random_rotation_3d(volume, max_angle=15)
        
        assert result.shape == volume.shape
    
    def test_augmentation_pipeline(self):
        """Test augmentation pipeline composition."""
        volume = np.random.rand(32, 32, 32).astype(np.float32)
        
        pipeline = AugmentationPipeline()
        pipeline.add_flip(prob=0.5)
        pipeline.add_rotation(max_angle=10, prob=0.5)
        
        result = pipeline(volume)
        
        assert result.shape == volume.shape
        assert result.dtype == np.float32


class TestBatchGenerator:
    """Tests for batch data generation."""
    
    def test_batch_generator_output_shape(self):
        """Test that batch generator produces correct shapes."""
        # This test requires actual data or mocking
        pass  # Skip for now - requires data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
