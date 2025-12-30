#!/bin/bash
# MRI Classification System - WSL2/Mojo Setup Script
# Run this script inside WSL2 Ubuntu

set -e

echo "============================================"
echo "MRI Classification System - Environment Setup"
echo "============================================"

# Update system packages
echo "[1/5] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and pip
echo "[2/5] Installing Python environment..."
sudo apt-get install -y python3 python3-pip python3-venv

# Create Python virtual environment
echo "[3/5] Creating Python virtual environment..."
cd /mnt/c/mri
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "[4/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Modular CLI and Mojo SDK
echo "[5/5] Installing Mojo SDK..."
curl -s https://get.modular.com | sh -

# Add Modular to PATH
echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.bashrc
echo 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install Mojo
modular install mojo

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  source /mnt/c/mri/.venv/bin/activate"
echo ""
echo "To verify Mojo installation:"
echo "  mojo --version"
echo ""
