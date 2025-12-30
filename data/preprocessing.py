"""
MRI Preprocessing Pipeline
==========================
Handles intensity normalization, resampling, and preprocessing transforms.
"""

from typing import Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom


def normalize_intensity(
    volume: np.ndarray,
    method: str = "zscore",
    clip_percentile: Optional[Tuple[float, float]] = (0.5, 99.5)
) -> np.ndarray:
    """
    Normalize voxel intensities.
    
    Args:
        volume: 3D numpy array
        method: Normalization method - 'zscore', 'minmax', or 'histogram'
        clip_percentile: Optional percentile clipping before normalization
        
    Returns:
        Normalized volume
    """
    volume = volume.astype(np.float32)
    
    # Clip outliers
    if clip_percentile is not None:
        p_low, p_high = np.percentile(volume, clip_percentile)
        volume = np.clip(volume, p_low, p_high)
    
    if method == "zscore":
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            volume = (volume - mean) / std
        else:
            volume = volume - mean
            
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(volume)
        max_val = np.max(volume)
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
        else:
            volume = volume - min_val
            
    elif method == "histogram":
        # Histogram equalization
        # Flatten, compute histogram, apply mapping
        flat = volume.flatten()
        hist, bin_edges = np.histogram(flat, bins=256, density=True)
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]  # Normalize
        
        # Map values
        bin_indices = np.digitize(flat, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, len(cdf) - 1)
        volume = cdf[bin_indices].reshape(volume.shape)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return volume.astype(np.float32)


def resample_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    current_spacing: Optional[Tuple[float, float, float]] = None,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    order: int = 1
) -> np.ndarray:
    """
    Resample volume to target shape or spacing.
    
    Args:
        volume: 3D numpy array
        target_shape: Target dimensions (D, H, W)
        current_spacing: Current voxel spacing in mm (optional)
        target_spacing: Target voxel spacing in mm (optional)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns:
        Resampled volume
    """
    current_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)
    
    if target_spacing is not None and current_spacing is not None:
        # Calculate zoom factors based on spacing
        current_spacing = np.array(current_spacing)
        target_spacing = np.array(target_spacing)
        zoom_factors = current_spacing / target_spacing
    else:
        # Calculate zoom factors based on shape
        zoom_factors = target_shape / current_shape
    
    # Apply zoom
    resampled = zoom(volume, zoom_factors, order=order, mode='nearest')
    
    # Ensure exact target shape (zoom can be slightly off)
    if resampled.shape != tuple(target_shape):
        resampled = _crop_or_pad(resampled, target_shape)
    
    return resampled.astype(np.float32)


def _crop_or_pad(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Crop or pad volume to exact target shape."""
    result = np.zeros(target_shape, dtype=volume.dtype)
    
    # Calculate crop/pad for each dimension
    for i in range(3):
        if volume.shape[i] > target_shape[i]:
            # Crop (center)
            start = (volume.shape[i] - target_shape[i]) // 2
            slc = slice(start, start + target_shape[i])
            if i == 0:
                volume = volume[slc, :, :]
            elif i == 1:
                volume = volume[:, slc, :]
            else:
                volume = volume[:, :, slc]
    
    # Calculate padding offsets
    pad_before = [(max(0, target_shape[i] - volume.shape[i]) // 2) for i in range(3)]
    
    # Place volume in result
    result[
        pad_before[0]:pad_before[0] + volume.shape[0],
        pad_before[1]:pad_before[1] + volume.shape[1],
        pad_before[2]:pad_before[2] + volume.shape[2]
    ] = volume
    
    return result


def extract_brain_mask(volume: np.ndarray, threshold_percentile: float = 10.0) -> np.ndarray:
    """
    Simple brain mask extraction based on intensity thresholding.
    For more accurate results, use specialized tools like FSL BET.
    
    Args:
        volume: 3D brain MRI volume
        threshold_percentile: Percentile for background threshold
        
    Returns:
        Binary mask array
    """
    threshold = np.percentile(volume[volume > 0], threshold_percentile)
    mask = volume > threshold
    
    # Fill holes and clean up
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=2)
    
    # Keep largest connected component
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest
    
    return mask.astype(np.float32)


def apply_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask to volume."""
    return volume * mask


@dataclass
class PreprocessingStep:
    """Single preprocessing step with parameters."""
    name: str
    fn: Callable
    kwargs: dict


class PreprocessingPipeline:
    """
    Composable preprocessing pipeline for MRI volumes.
    
    Example:
        pipeline = PreprocessingPipeline()
        pipeline.add_normalize('zscore')
        pipeline.add_resample((64, 64, 64))
        
        processed = pipeline(volume)
    """
    
    def __init__(self):
        self.steps: List[PreprocessingStep] = []
    
    def add_step(self, name: str, fn: Callable, **kwargs):
        """Add a custom preprocessing step."""
        self.steps.append(PreprocessingStep(name, fn, kwargs))
        return self
    
    def add_normalize(
        self,
        method: str = "zscore",
        clip_percentile: Optional[Tuple[float, float]] = (0.5, 99.5)
    ):
        """Add intensity normalization step."""
        self.steps.append(PreprocessingStep(
            name="normalize",
            fn=normalize_intensity,
            kwargs={'method': method, 'clip_percentile': clip_percentile}
        ))
        return self
    
    def add_resample(
        self,
        target_shape: Tuple[int, int, int],
        order: int = 1
    ):
        """Add resampling step."""
        self.steps.append(PreprocessingStep(
            name="resample",
            fn=resample_volume,
            kwargs={'target_shape': target_shape, 'order': order}
        ))
        return self
    
    def add_mask_extraction(self, threshold_percentile: float = 10.0):
        """Add brain mask extraction and application."""
        def mask_and_apply(volume, threshold_percentile):
            mask = extract_brain_mask(volume, threshold_percentile)
            return apply_mask(volume, mask)
        
        self.steps.append(PreprocessingStep(
            name="mask",
            fn=mask_and_apply,
            kwargs={'threshold_percentile': threshold_percentile}
        ))
        return self
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps in order."""
        result = volume.copy()
        for step in self.steps:
            result = step.fn(result, **step.kwargs)
        return result
    
    def __repr__(self) -> str:
        steps_str = " -> ".join(s.name for s in self.steps)
        return f"PreprocessingPipeline({steps_str})"


def create_default_pipeline(
    target_shape: Tuple[int, int, int] = (64, 64, 64),
    normalization: str = "zscore"
) -> PreprocessingPipeline:
    """
    Create a standard preprocessing pipeline.
    
    Args:
        target_shape: Target volume dimensions
        normalization: Normalization method
        
    Returns:
        Configured PreprocessingPipeline
    """
    pipeline = PreprocessingPipeline()
    pipeline.add_normalize(method=normalization)
    pipeline.add_resample(target_shape)
    return pipeline
