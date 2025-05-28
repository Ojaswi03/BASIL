#!/bin/bash

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo " Creating Python virtual environment 'venv-basil'..."
python3 -m venv venv-basil

echo "Virtual environment created."
echo "   To activate it, run: source venv-basil/bin/activate"

echo " Installing required Python packages..."
source venv-basil/bin/activate

pip install --upgrade pip
pip install numpy matplotlib wandb tensorflow scikit-learn torch torchvision pandas

echo ""
echo "All dependencies installed successfully."
echo "   You can now run your project using: source venv-basil/bin/activate && python main_basil.py"
