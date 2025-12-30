"""
3D Data Augmentation for MRI Volumes
====================================
Implements 3D-specific augmentation techniques for medical imaging.
"""

from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.ndimage import rotate, shift, zoom


def random_flip_3d(
    volume: np.ndarray,
    axis: Optional[int] = None,
    prob: float = 0.5
) -> np.ndarray:
    """
    Randomly flip volume along specified axis.
    
    Args:
        volume: 3D numpy array
        axis: Axis to flip (0=sagittal, 1=coronal, 2=axial). If None, random axis.
        prob: Probability of flipping
        
    Returns:
        Possibly flipped volume
    """
    if np.random.random() > prob:
        return volume
    
    if axis is None:
        axis = np.random.randint(0, 3)
    
    return np.flip(volume, axis=axis).copy()


def random_rotation_3d(
    volume: np.ndarray,
    max_angle: float = 15.0,
    axes: Optional[Tuple[int, int]] = None,
    order: int = 1
) -> np.ndarray:
    """
    Randomly rotate volume around specified axes.
    
    Args:
        volume: 3D numpy array
        max_angle: Maximum rotation angle in degrees
        axes: Rotation plane as (axis1, axis2). If None, random plane.
        order: Interpolation order
        
    Returns:
        Rotated volume
    """
    angle = np.random.uniform(-max_angle, max_angle)
    
    if axes is None:
        # Random rotation plane
        axes = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
    
    rotated = rotate(volume, angle, axes=axes, reshape=False, order=order, mode='nearest')
    return rotated.astype(np.float32)


def random_shift_3d(
    volume: np.ndarray,
    max_shift: float = 0.1,
    order: int = 1
) -> np.ndarray:
    """
    Randomly translate volume.
    
    Args:
        volume: 3D numpy array
        max_shift: Maximum shift as fraction of dimension size
        order: Interpolation order
        
    Returns:
        Shifted volume
    """
    shifts = [
        np.random.uniform(-max_shift, max_shift) * dim
        for dim in volume.shape
    ]
    
    shifted = shift(volume, shifts, order=order, mode='nearest')
    return shifted.astype(np.float32)


def random_zoom_3d(
    volume: np.ndarray,
    zoom_range: Tuple[float, float] = (0.9, 1.1),
    order: int = 1
) -> np.ndarray:
    """
    Randomly zoom volume (scale).
    
    Args:
        volume: 3D numpy array
        zoom_range: (min_zoom, max_zoom) range
        order: Interpolation order
        
    Returns:
        Zoomed volume (same size via crop/pad)
    """
    original_shape = volume.shape
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    
    # Apply zoom
    zoomed = zoom(volume, zoom_factor, order=order, mode='nearest')
    
    # Crop or pad back to original shape
    result = np.zeros(original_shape, dtype=np.float32)
    
    # Calculate center crop/pad
    for axis in range(3):
        if zoomed.shape[axis] > original_shape[axis]:
            # Crop
            start = (zoomed.shape[axis] - original_shape[axis]) // 2
            slc = [slice(None)] * 3
            slc[axis] = slice(start, start + original_shape[axis])
            zoomed = zoomed[tuple(slc)]
    
    # Pad if needed
    pad_before = [(original_shape[i] - zoomed.shape[i]) // 2 for i in range(3)]
    result[
        pad_before[0]:pad_before[0] + zoomed.shape[0],
        pad_before[1]:pad_before[1] + zoomed.shape[1],
        pad_before[2]:pad_before[2] + zoomed.shape[2]
    ] = zoomed
    
    return result


def elastic_deformation_3d(
    volume: np.ndarray,
    alpha: float = 100.0,
    sigma: float = 10.0,
    order: int = 1
) -> np.ndarray:
    """
    Apply elastic deformation to volume.
    Simulates tissue deformation common in medical imaging.
    
    Args:
        volume: 3D numpy array
        alpha: Deformation intensity
        sigma: Smoothing sigma for deformation field
        order: Interpolation order
        
    Returns:
        Deformed volume
    """
    shape = volume.shape
    
    # Generate random displacement fields
    dx = ndimage.gaussian_filter(
        (np.random.random(shape) * 2 - 1), sigma
    ) * alpha
    dy = ndimage.gaussian_filter(
        (np.random.random(shape) * 2 - 1), sigma
    ) * alpha
    dz = ndimage.gaussian_filter(
        (np.random.random(shape) * 2 - 1), sigma
    ) * alpha
    
    # Create coordinate grids
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    
    # Apply deformation
    indices = [
        np.clip(z + dz, 0, shape[0] - 1),
        np.clip(y + dy, 0, shape[1] - 1),
        np.clip(x + dx, 0, shape[2] - 1)
    ]
    
    deformed = ndimage.map_coordinates(volume, indices, order=order, mode='nearest')
    return deformed.astype(np.float32)


def random_intensity_shift(
    volume: np.ndarray,
    shift_range: float = 0.1,
    scale_range: Tuple[float, float] = (0.9, 1.1)
) -> np.ndarray:
    """
    Randomly shift and scale intensities.
    
    Args:
        volume: 3D numpy array
        shift_range: Maximum intensity shift
        scale_range: Intensity scale range
        
    Returns:
        Intensity-modified volume
    """
    shift_val = np.random.uniform(-shift_range, shift_range)
    scale_val = np.random.uniform(scale_range[0], scale_range[1])
    
    return (volume * scale_val + shift_val).astype(np.float32)


def random_noise(
    volume: np.ndarray,
    noise_std: float = 0.05
) -> np.ndarray:
    """
    Add random Gaussian noise.
    
    Args:
        volume: 3D numpy array
        noise_std: Standard deviation of noise
        
    Returns:
        Noisy volume
    """
    noise = np.random.normal(0, noise_std, volume.shape)
    return (volume + noise).astype(np.float32)


@dataclass
class AugmentationStep:
    """Single augmentation step with parameters."""
    name: str
    fn: Callable
    prob: float
    kwargs: dict


class AugmentationPipeline:
    """
    Composable augmentation pipeline for 3D volumes.
    Each augmentation has an independent probability of being applied.
    
    Example:
        aug = AugmentationPipeline()
        aug.add_flip(prob=0.5)
        aug.add_rotation(max_angle=15, prob=0.5)
        aug.add_elastic(alpha=100, sigma=10, prob=0.3)
        
        augmented = aug(volume)
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.steps: List[AugmentationStep] = []
        if seed is not None:
            np.random.seed(seed)
    
    def add_flip(self, axis: Optional[int] = None, prob: float = 0.5):
        """Add random flip augmentation."""
        self.steps.append(AugmentationStep(
            name="flip",
            fn=random_flip_3d,
            prob=prob,
            kwargs={'axis': axis, 'prob': 1.0}  # prob handled externally
        ))
        return self
    
    def add_rotation(self, max_angle: float = 15.0, prob: float = 0.5):
        """Add random rotation augmentation."""
        self.steps.append(AugmentationStep(
            name="rotation",
            fn=random_rotation_3d,
            prob=prob,
            kwargs={'max_angle': max_angle}
        ))
        return self
    
    def add_shift(self, max_shift: float = 0.1, prob: float = 0.5):
        """Add random translation augmentation."""
        self.steps.append(AugmentationStep(
            name="shift",
            fn=random_shift_3d,
            prob=prob,
            kwargs={'max_shift': max_shift}
        ))
        return self
    
    def add_zoom(self, zoom_range: Tuple[float, float] = (0.9, 1.1), prob: float = 0.5):
        """Add random scaling augmentation."""
        self.steps.append(AugmentationStep(
            name="zoom",
            fn=random_zoom_3d,
            prob=prob,
            kwargs={'zoom_range': zoom_range}
        ))
        return self
    
    def add_elastic(self, alpha: float = 100.0, sigma: float = 10.0, prob: float = 0.3):
        """Add elastic deformation augmentation."""
        self.steps.append(AugmentationStep(
            name="elastic",
            fn=elastic_deformation_3d,
            prob=prob,
            kwargs={'alpha': alpha, 'sigma': sigma}
        ))
        return self
    
    def add_intensity(
        self,
        shift_range: float = 0.1,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.5
    ):
        """Add random intensity shift/scale augmentation."""
        self.steps.append(AugmentationStep(
            name="intensity",
            fn=random_intensity_shift,
            prob=prob,
            kwargs={'shift_range': shift_range, 'scale_range': scale_range}
        ))
        return self
    
    def add_noise(self, noise_std: float = 0.05, prob: float = 0.3):
        """Add random noise augmentation."""
        self.steps.append(AugmentationStep(
            name="noise",
            fn=random_noise,
            prob=prob,
            kwargs={'noise_std': noise_std}
        ))
        return self
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Apply augmentations with respective probabilities."""
        result = volume.copy()
        for step in self.steps:
            if np.random.random() < step.prob:
                result = step.fn(result, **step.kwargs)
        return result
    
    def __repr__(self) -> str:
        steps_str = ", ".join(f"{s.name}(p={s.prob})" for s in self.steps)
        return f"AugmentationPipeline([{steps_str}])"


def create_default_augmentation(
    flip_prob: float = 0.5,
    rotation_prob: float = 0.5,
    elastic_prob: float = 0.3
) -> AugmentationPipeline:
    """
    Create a standard augmentation pipeline for brain MRI.
    
    Args:
        flip_prob: Probability of random flip
        rotation_prob: Probability of random rotation
        elastic_prob: Probability of elastic deformation
        
    Returns:
        Configured AugmentationPipeline
    """
    pipeline = AugmentationPipeline()
    pipeline.add_flip(axis=2, prob=flip_prob)  # Left-right flip common for brain
    pipeline.add_rotation(max_angle=15, prob=rotation_prob)
    pipeline.add_intensity(prob=0.3)
    pipeline.add_elastic(alpha=50, sigma=8, prob=elastic_prob)
    return pipeline
