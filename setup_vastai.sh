#!/bin/bash
# ================================================
# vast.ai Setup Script for MRI Classification
# ================================================
# Run this script on your vast.ai instance after cloning the repo
# Usage: chmod +x setup_vastai.sh && ./setup_vastai.sh

set -e

echo "============================================"
echo "MRI Classification - vast.ai Setup"
echo "============================================"

# Update system
echo "[1/6] Updating system packages..."
apt-get update && apt-get upgrade -y

# Install Python and essentials
echo "[2/6] Installing Python and build tools..."
apt-get install -y python3 python3-pip python3-venv build-essential curl git

# Create virtual environment
echo "[3/6] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Modular CLI and Mojo
echo "[5/6] Installing Mojo SDK..."
curl -s https://get.modular.com | sh -

# Add Modular to PATH for current session
export MODULAR_HOME="$HOME/.modular"
export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"

# Install Mojo
modular install mojo

# Add to bashrc for persistence
echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.bashrc
echo 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.bashrc

# Create directories
echo "[6/6] Setting up project directories..."
mkdir -p data/raw data/processed checkpoints logs

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify installations:"
echo "  python3 --version"
echo "  mojo --version"
echo ""
echo "To start training:"
echo "  mojo run scripts/train.mojo"
echo ""
echo "Place your MRI data in data/raw/<class_name>/*.nii.gz"
echo "============================================"
