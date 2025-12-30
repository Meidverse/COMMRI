#!/bin/bash
# ================================================
# vast.ai Setup Script for MRI Classification
# ================================================
# Run this script on your vast.ai instance after cloning the repo
# Usage: chmod +x setup_vastai.sh && ./setup_vastai.sh
#
# Options:
#   --skip-data     Skip synthetic data generation
#   --with-ixi      Also download IXI dataset (large, ~5GB)
#   --with-kaggle   Also download Kaggle tumor dataset (requires API key)
#   --skip-mojo     Skip Mojo installation (use Python-only mode)

set -e

# Parse arguments
SKIP_DATA=false
WITH_IXI=false
WITH_KAGGLE=false
SKIP_MOJO=false

for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --with-ixi) WITH_IXI=true ;;
        --with-kaggle) WITH_KAGGLE=true ;;
        --skip-mojo) SKIP_MOJO=true ;;
    esac
done

echo "============================================"
echo "MRI Classification - vast.ai Setup"
echo "============================================"
echo "Time: $(date)"
echo ""

# Update system
echo "[1/7] Updating system packages..."
apt-get update && apt-get upgrade -y

# Install Python and essentials
echo "[2/7] Installing Python and build tools..."
apt-get install -y python3 python3-pip python3-venv build-essential curl git wget

# Create virtual environment
echo "[3/7] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "[4/7] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Mojo SDK (optional - can run in Python-only mode)
if [ "$SKIP_MOJO" = false ]; then
    echo "[5/7] Installing Mojo SDK..."
    curl -s https://get.modular.com | sh -
    
    # IMPORTANT: Source the modular paths BEFORE using modular command
    export MODULAR_HOME="$HOME/.modular"
    export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"
    
    # Also need to source the new shell config that modular creates
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc" 2>/dev/null || true
    fi
    
    # Try to install mojo
    if command -v modular &> /dev/null; then
        modular install mojo
        echo "Mojo installed successfully!"
    else
        echo "WARNING: modular command not found in PATH"
        echo "Trying alternative installation..."
        # The modular installer may have put it somewhere else
        if [ -f "$HOME/.modular/bin/modular" ]; then
            export PATH="$HOME/.modular/bin:$PATH"
            "$HOME/.modular/bin/modular" install mojo
        else
            echo "Mojo installation failed - will use Python-only mode"
            SKIP_MOJO=true
        fi
    fi
    
    # Add to bashrc for persistence
    echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.bashrc
    echo 'export PATH="$HOME/.modular/bin:$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.bashrc
else
    echo "[5/7] Skipping Mojo installation (--skip-mojo flag set)"
fi

# Create directories
echo "[6/7] Setting up project directories..."
mkdir -p data/raw/healthy data/raw/diseased data/processed checkpoints logs outputs

# Download/generate data
echo "[7/7] Preparing training data..."
if [ "$SKIP_DATA" = false ]; then
    echo "Generating synthetic training data..."
    python3 scripts/download_data.py --dataset synthetic --output ./data/raw --num-samples 50
    
    if [ "$WITH_IXI" = true ]; then
        echo "Downloading IXI dataset..."
        python3 scripts/download_data.py --dataset ixi --modality T1 --output ./data/raw
    fi
    
    if [ "$WITH_KAGGLE" = true ]; then
        echo "Downloading Kaggle tumor dataset..."
        python3 scripts/download_data.py --dataset kaggle-tumor --output ./data/raw
    fi
else
    echo "Skipping data download (--skip-data flag set)"
fi

echo ""
echo "============================================"
echo "         Setup Complete!                   "
echo "============================================"
echo ""
echo "Environment activation:"
echo "  source .venv/bin/activate"
echo ""
echo "Verify installations:"
echo "  python3 --version"
if [ "$SKIP_MOJO" = false ]; then
    echo "  mojo --version"
fi
echo ""
echo "Available data:"
ls -la data/raw/ 2>/dev/null || echo "  No data yet"
echo ""
echo "Download more data:"
echo "  python3 scripts/download_data.py --info"
echo ""
echo "Start training:"
if [ "$SKIP_MOJO" = false ]; then
    echo "  mojo run scripts/train.mojo          # Mojo (faster)"
fi
echo "  python run_training.py --data_dir ./data/raw  # Python"
echo ""
echo "============================================"
