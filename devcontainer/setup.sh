#!/bin/bash

echo "🚀 Setting up Federated Learning + ZKP Environment..."

# 1. Update System
sudo apt-get update && sudo apt-get install -y build-essential git curl

# 2. Install Node.js (Required for Circom and SnarkJS)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# 3. Install Circom and SnarkJS globally
npm install -g circom
npm install -g snarkjs

# 4. Install Python dependencies
pip install --upgrade pip
pip install "flwr[simulation]==1.13.0"
pip install opacus
pip install torch torchvision torchaudio
pip install ray
pip install numpy pandas scikit-learn

# 5. Verify Installation
echo "✅ Verification:"
node --version
npm --version
circom --version
python --version

echo "🎉 Setup Complete! You can now run your code."