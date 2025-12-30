#!/usr/bin/env python3
"""
MRI Dataset Download Script
============================
Downloads publicly available brain MRI datasets for training.

Supported Datasets:
1. IXI Dataset - Brain MRI from healthy subjects
2. OASIS-1 - Cross-sectional MRI (requires registration)
3. Kaggle Brain Tumor MRI Dataset (requires Kaggle API)

Usage:
    python download_data.py --dataset <dataset_name> --output <path>
    
Examples:
    python download_data.py --dataset ixi --output ./data/raw
    python download_data.py --dataset kaggle-tumor --output ./data/raw
    python download_data.py --all --output ./data/raw
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import Optional
import hashlib


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress indication."""
    try:
        print(f"\n{desc}: {url}")
        print(f"Destination: {dest_path}")
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                bar_len = 40
                filled = int(bar_len * percent / 100)
                bar = '=' * filled + '-' * (bar_len - filled)
                mb_done = count * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r[{bar}] {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)", end='', flush=True)
            else:
                mb_done = count * block_size / (1024 * 1024)
                print(f"\rDownloaded: {mb_done:.1f} MB", end='', flush=True)
        
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract zip, tar, tar.gz, or gz archives."""
    try:
        print(f"Extracting: {archive_path}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = ''.join(archive_path.suffixes).lower()
        
        if suffix.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(dest_dir)
        elif suffix.endswith('.tar.gz') or suffix.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(dest_dir)
        elif suffix.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(dest_dir)
        elif suffix.endswith('.gz'):
            out_path = dest_dir / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(out_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"Unknown archive format: {suffix}")
            return False
            
        print(f"Extracted to: {dest_dir}")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def organize_for_training(data_dir: Path, class_name: str = "healthy"):
    """Organize downloaded data into class directories."""
    class_dir = data_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all NIfTI files and move them to class directory
    nifti_files = list(data_dir.glob("**/*.nii.gz")) + list(data_dir.glob("**/*.nii"))
    
    for nifti_file in nifti_files:
        if nifti_file.parent != class_dir:
            dest = class_dir / nifti_file.name
            if not dest.exists():
                shutil.move(str(nifti_file), str(dest))
                print(f"Moved: {nifti_file.name} -> {class_name}/")


# ==============================================================================
# Dataset Downloaders
# ==============================================================================

def download_ixi_dataset(output_dir: Path, modality: str = "T1") -> bool:
    """
    Download IXI Dataset (Brain Development).
    
    The IXI Dataset contains ~600 MR images from normal, healthy subjects.
    Available modalities: T1, T2, PD, MRA, DTI
    
    Website: https://brain-development.org/ixi-dataset/
    """
    print("\n" + "="*60)
    print("IXI Dataset - Brain MRI from Healthy Subjects")
    print("="*60)
    print(f"Modality: {modality}")
    print("License: CC BY-SA 3.0")
    
    # IXI dataset URLs (sample subset for demo - full dataset is larger)
    base_url = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"
    
    modality_urls = {
        "T1": f"{base_url}/IXI-T1.tar",
        "T2": f"{base_url}/IXI-T2.tar", 
        "PD": f"{base_url}/IXI-PD.tar",
        "MRA": f"{base_url}/IXI-MRA.tar",
    }
    
    if modality not in modality_urls:
        print(f"Unknown modality: {modality}. Available: {list(modality_urls.keys())}")
        return False
    
    url = modality_urls[modality]
    archive_path = output_dir / f"IXI-{modality}.tar"
    
    if download_file(url, archive_path, f"Downloading IXI {modality}"):
        if extract_archive(archive_path, output_dir / "ixi"):
            organize_for_training(output_dir / "ixi", "healthy")
            print(f"\n✓ IXI {modality} dataset ready at: {output_dir}/ixi/")
            return True
    return False


def download_kaggle_tumor_dataset(output_dir: Path) -> bool:
    """
    Download Brain Tumor MRI Dataset from Kaggle.
    
    Requires Kaggle API credentials (~/.kaggle/kaggle.json)
    Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
    """
    print("\n" + "="*60)
    print("Kaggle Brain Tumor MRI Dataset")
    print("="*60)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except ImportError:
        print("Kaggle API not installed. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
        except Exception as e:
            print(f"Error: Could not authenticate with Kaggle API.")
            print("Please ensure ~/.kaggle/kaggle.json exists with valid credentials.")
            print("Get your API key from: https://www.kaggle.com/settings")
            return False
    except Exception as e:
        print(f"Kaggle authentication error: {e}")
        print("\nTo use Kaggle datasets:")
        print("1. Create account at https://www.kaggle.com")
        print("2. Go to Settings -> API -> Create New Token")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        return False
    
    try:
        dataset = "masoudnickparvar/brain-tumor-mri-dataset"
        print(f"Downloading: {dataset}")
        
        download_path = output_dir / "kaggle_tumor"
        download_path.mkdir(parents=True, exist_ok=True)
        
        api.dataset_download_files(dataset, path=str(download_path), unzip=True)
        
        print(f"\n✓ Brain Tumor dataset ready at: {download_path}/")
        print("\nDataset structure:")
        print("  - Training/glioma/")
        print("  - Training/meningioma/") 
        print("  - Training/notumor/")
        print("  - Training/pituitary/")
        return True
        
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False


def download_sample_data(output_dir: Path) -> bool:
    """
    Download sample NIfTI data for testing.
    Uses nibabel's sample data.
    """
    print("\n" + "="*60)
    print("Sample NIfTI Data for Testing")
    print("="*60)
    
    try:
        import nibabel as nib
        from nibabel import testing
        
        # Create sample directory
        sample_dir = output_dir / "sample"
        (sample_dir / "healthy").mkdir(parents=True, exist_ok=True)
        (sample_dir / "diseased").mkdir(parents=True, exist_ok=True)
        
        # Get sample NIfTI file path from nibabel
        sample_file = Path(testing.data_path) / "example4d.nii.gz"
        
        if sample_file.exists():
            # Load and create synthetic samples
            img = nib.load(str(sample_file))
            
            # Create a few synthetic samples for each class
            import numpy as np
            for i in range(3):
                # Healthy samples (original)
                healthy_path = sample_dir / "healthy" / f"sample_healthy_{i:03d}.nii.gz"
                if not healthy_path.exists():
                    if len(img.shape) == 4:
                        data = img.get_fdata()[:, :, :, 0]
                    else:
                        data = img.get_fdata()
                    new_img = nib.Nifti1Image(data.astype(np.float32), img.affine)
                    nib.save(new_img, str(healthy_path))
                    print(f"Created: {healthy_path.name}")
                
                # Diseased samples (with synthetic "lesion")
                diseased_path = sample_dir / "diseased" / f"sample_diseased_{i:03d}.nii.gz"
                if not diseased_path.exists():
                    if len(img.shape) == 4:
                        data = img.get_fdata()[:, :, :, 0].copy()
                    else:
                        data = img.get_fdata().copy()
                    # Add synthetic lesion (bright spot)
                    cx, cy, cz = np.array(data.shape) // 2
                    r = min(data.shape) // 8
                    for x in range(max(0, cx-r), min(data.shape[0], cx+r)):
                        for y in range(max(0, cy-r), min(data.shape[1], cy+r)):
                            for z in range(max(0, cz-r), min(data.shape[2], cz+r)):
                                if (x-cx)**2 + (y-cy)**2 + (z-cz)**2 < r**2:
                                    data[x, y, z] = data.max() * 1.5
                    new_img = nib.Nifti1Image(data.astype(np.float32), img.affine)
                    nib.save(new_img, str(diseased_path))
                    print(f"Created: {diseased_path.name}")
            
            print(f"\n✓ Sample data ready at: {sample_dir}/")
            return True
        else:
            print("nibabel sample data not found.")
            return False
            
    except ImportError:
        print("nibabel not installed. Run: pip install nibabel")
        return False
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return False


def download_synthetic_data(output_dir: Path, num_samples: int = 20) -> bool:
    """
    Generate synthetic 3D brain MRI-like data for testing.
    No external downloads required.
    """
    print("\n" + "="*60)
    print("Generating Synthetic Brain MRI Data")
    print("="*60)
    print(f"Number of samples per class: {num_samples}")
    
    try:
        import numpy as np
        import nibabel as nib
        
        synth_dir = output_dir / "synthetic"
        (synth_dir / "healthy").mkdir(parents=True, exist_ok=True)
        (synth_dir / "diseased").mkdir(parents=True, exist_ok=True)
        
        shape = (64, 64, 64)
        affine = np.eye(4)
        
        for i in range(num_samples):
            np.random.seed(i)
            
            # Create brain-like volume (ellipsoid with noise)
            x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
            cx, cy, cz = np.array(shape) // 2
            
            # Brain mask (ellipsoid)
            brain_mask = ((x-cx)/20)**2 + ((y-cy)/25)**2 + ((z-cz)/20)**2 < 1
            
            # Base brain tissue with realistic intensity variations
            base_intensity = np.random.uniform(0.4, 0.6)
            brain = np.zeros(shape, dtype=np.float32)
            brain[brain_mask] = base_intensity
            
            # Add ventricle-like structures (darker)
            v_mask = ((x-cx)/5)**2 + ((y-cy)/8)**2 + ((z-cz)/5)**2 < 1
            brain[v_mask] = 0.2
            
            # Add cortex-like variations
            brain += np.random.randn(*shape).astype(np.float32) * 0.05
            brain = np.clip(brain, 0, 1)
            
            # Save healthy sample
            healthy_path = synth_dir / "healthy" / f"brain_healthy_{i:03d}.nii.gz"
            img = nib.Nifti1Image(brain, affine)
            nib.save(img, str(healthy_path))
            
            # Create diseased version with lesion
            diseased = brain.copy()
            
            # Add tumor/lesion
            lx = np.random.randint(cx-10, cx+10)
            ly = np.random.randint(cy-10, cy+10)
            lz = np.random.randint(cz-10, cz+10)
            lr = np.random.randint(3, 8)
            
            lesion_mask = ((x-lx)/lr)**2 + ((y-ly)/lr)**2 + ((z-lz)/lr)**2 < 1
            diseased[lesion_mask] = np.random.uniform(0.8, 1.0)
            
            # Add edema around lesion
            edema_mask = ((x-lx)/(lr*1.5))**2 + ((y-ly)/(lr*1.5))**2 + ((z-lz)/(lr*1.5))**2 < 1
            diseased[edema_mask & ~lesion_mask] = np.random.uniform(0.6, 0.75)
            
            diseased_path = synth_dir / "diseased" / f"brain_diseased_{i:03d}.nii.gz"
            img = nib.Nifti1Image(diseased, affine)
            nib.save(img, str(diseased_path))
            
            print(f"Generated sample {i+1}/{num_samples}", end='\r')
        
        print(f"\n\n✓ Synthetic data generated at: {synth_dir}/")
        print(f"  - healthy/: {num_samples} samples")
        print(f"  - diseased/: {num_samples} samples")
        return True
        
    except ImportError:
        print("Required libraries not installed. Run: pip install numpy nibabel")
        return False
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_dataset_info():
    """Print information about available datasets."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                   MRI Dataset Download Options                              ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  1. SYNTHETIC (--dataset synthetic)                                         ║
║     • Generated locally, no download needed                                 ║
║     • 20 samples per class (healthy/diseased)                               ║
║     • Perfect for testing and development                                   ║
║                                                                             ║
║  2. IXI Dataset (--dataset ixi)                                             ║
║     • ~600 brain MRIs from healthy subjects                                 ║
║     • T1, T2, PD, MRA modalities available                                  ║
║     • License: CC BY-SA 3.0                                                 ║
║     • Download size: ~5-10GB per modality                                   ║
║                                                                             ║
║  3. Kaggle Brain Tumor (--dataset kaggle-tumor)                             ║
║     • Brain tumor classification dataset                                    ║
║     • Classes: glioma, meningioma, notumor, pituitary                       ║
║     • Requires Kaggle API credentials                                       ║
║                                                                             ║
║  4. Sample Data (--dataset sample)                                          ║
║     • Uses nibabel's built-in test data                                     ║
║     • Creates synthetic healthy/diseased samples                            ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Usage Examples:
    # Generate synthetic data for quick testing
    python download_data.py --dataset synthetic --output ./data/raw
    
    # Download IXI T1 scans
    python download_data.py --dataset ixi --modality T1 --output ./data/raw
    
    # Download Kaggle brain tumor dataset
    python download_data.py --dataset kaggle-tumor --output ./data/raw
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download MRI datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py --dataset synthetic --output ./data/raw
    python download_data.py --dataset ixi --modality T1 --output ./data/raw
    python download_data.py --dataset kaggle-tumor --output ./data/raw
    python download_data.py --info
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        choices=["ixi", "kaggle-tumor", "sample", "synthetic"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data/raw"),
        help="Output directory (default: ./data/raw)"
    )
    parser.add_argument(
        "--modality", "-m",
        choices=["T1", "T2", "PD", "MRA"],
        default="T1",
        help="Modality for IXI dataset (default: T1)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=20,
        help="Number of samples per class for synthetic data (default: 20)"
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Show information about available datasets"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets"
    )
    
    args = parser.parse_args()
    
    if args.info or (not args.dataset and not args.all):
        print_dataset_info()
        return 0
    
    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    if args.all:
        # Download all datasets
        success = success and download_synthetic_data(args.output, args.num_samples)
        success = success and download_sample_data(args.output)
        success = success and download_ixi_dataset(args.output, args.modality)
        # Kaggle requires auth, try but don't fail overall
        download_kaggle_tumor_dataset(args.output)
    else:
        # Download specific dataset
        if args.dataset == "ixi":
            success = download_ixi_dataset(args.output, args.modality)
        elif args.dataset == "kaggle-tumor":
            success = download_kaggle_tumor_dataset(args.output)
        elif args.dataset == "sample":
            success = download_sample_data(args.output)
        elif args.dataset == "synthetic":
            success = download_synthetic_data(args.output, args.num_samples)
    
    if success:
        print("\n" + "="*60)
        print(" Download Complete! ")
        print("="*60)
        print(f"\nData location: {args.output.absolute()}")
        print("\nNext steps:")
        print("  1. Review the downloaded data")
        print("  2. Update config/config.yaml with data paths")
        print("  3. Run training: mojo run scripts/train.mojo")
        print("="*60)
        return 0
    else:
        print("\n[!] Some downloads failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
