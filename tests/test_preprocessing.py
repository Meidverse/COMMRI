"""
Test suite for the preprocessing module.
Run with: python -m pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import (
    normalize_intensity,
    resample_volume,
    extract_brain_mask,
    apply_mask,
    PreprocessingPipeline,
    create_default_pipeline
)


class TestNormalization:
    """Tests for intensity normalization."""
    
    def test_zscore_normalization_statistics(self):
        """Verify z-score produces mean~0, std~1."""
        np.random.seed(42)
        volume = np.random.randn(50, 50, 50).astype(np.float32) * 100 + 500
        
        result = normalize_intensity(volume, method="zscore", clip_percentile=None)
        
        assert abs(result.mean()) < 0.01
        assert abs(result.std() - 1.0) < 0.01
    
    def test_minmax_normalization_range(self):
        """Verify min-max produces values in [0, 1]."""
        np.random.seed(42)
        volume = np.random.rand(50, 50, 50).astype(np.float32) * 1000 - 200
        
        result = normalize_intensity(volume, method="minmax", clip_percentile=None)
        
        assert result.min() >= -0.001
        assert result.max() <= 1.001
    
    def test_percentile_clipping(self):
        """Test that percentile clipping reduces outliers."""
        volume = np.random.rand(50, 50, 50).astype(np.float32)
        volume[0, 0, 0] = 1000  # Outlier
        volume[1, 1, 1] = -500  # Outlier
        
        result = normalize_intensity(volume, method="zscore", clip_percentile=(1, 99))
        
        # Outliers should be clipped
        assert result.max() < 50  # Much smaller than original outlier
    
    def test_constant_volume(self):
        """Test handling of constant volume (std=0)."""
        volume = np.ones((10, 10, 10), dtype=np.float32) * 42
        
        result = normalize_intensity(volume, method="zscore")
        
        # Should not produce NaN or inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestResampling:
    """Tests for volume resampling."""
    
    def test_downsample(self):
        """Test downsampling to smaller dimensions."""
        volume = np.random.rand(64, 64, 64).astype(np.float32)
        target = (32, 32, 32)
        
        result = resample_volume(volume, target)
        
        assert result.shape == target
    
    def test_upsample(self):
        """Test upsampling to larger dimensions."""
        volume = np.random.rand(32, 32, 32).astype(np.float32)
        target = (64, 64, 64)
        
        result = resample_volume(volume, target)
        
        assert result.shape == target
    
    def test_non_cubic(self):
        """Test resampling non-cubic volumes."""
        volume = np.random.rand(40, 60, 80).astype(np.float32)
        target = (64, 64, 64)
        
        result = resample_volume(volume, target)
        
        assert result.shape == target
    
    def test_interpolation_order(self):
        """Test different interpolation orders produce results."""
        volume = np.random.rand(32, 32, 32).astype(np.float32)
        target = (64, 64, 64)
        
        # Nearest neighbor
        result_nn = resample_volume(volume, target, order=0)
        # Linear
        result_lin = resample_volume(volume, target, order=1)
        # Cubic
        result_cubic = resample_volume(volume, target, order=3)
        
        assert result_nn.shape == target
        assert result_lin.shape == target
        assert result_cubic.shape == target


class TestBrainMask:
    """Tests for brain mask extraction."""
    
    def test_mask_binary(self):
        """Test that mask is binary."""
        volume = np.random.rand(50, 50, 50).astype(np.float32)
        
        mask = extract_brain_mask(volume)
        
        unique_vals = np.unique(mask)
        assert len(unique_vals) <= 2
        assert all(v in [0, 1] for v in unique_vals)
    
    def test_apply_mask(self):
        """Test mask application zeros out background."""
        volume = np.ones((10, 10, 10), dtype=np.float32) * 100
        mask = np.zeros((10, 10, 10), dtype=np.float32)
        mask[3:7, 3:7, 3:7] = 1.0  # Small cube in center
        
        result = apply_mask(volume, mask)
        
        assert result[0, 0, 0] == 0  # Outside mask
        assert result[5, 5, 5] == 100  # Inside mask


class TestPipeline:
    """Tests for preprocessing pipeline."""
    
    def test_pipeline_composition(self):
        """Test multiple steps in pipeline."""
        volume = np.random.rand(64, 64, 64).astype(np.float32) * 1000
        
        pipeline = PreprocessingPipeline()
        pipeline.add_normalize("zscore")
        pipeline.add_resample((32, 32, 32))
        
        result = pipeline(volume)
        
        assert result.shape == (32, 32, 32)
    
    def test_default_pipeline(self):
        """Test default pipeline factory."""
        volume = np.random.rand(80, 80, 80).astype(np.float32) * 500
        
        pipeline = create_default_pipeline(
            target_shape=(64, 64, 64),
            normalization="minmax"
        )
        
        result = pipeline(volume)
        
        assert result.shape == (64, 64, 64)
    
    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        pipeline = PreprocessingPipeline()
        pipeline.add_normalize()
        pipeline.add_resample((32, 32, 32))
        
        repr_str = repr(pipeline)
        
        assert "normalize" in repr_str
        assert "resample" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
