#!/bin/bash
#SBATCH --job-name=setup_ctrlg_env
#SBATCH --output=setup_ctrlg_env_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

# 1. Initialize Conda from your home directory
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Create a clean environment (Bare-bones to avoid Conda solver hangs)
echo "Removing old environment (if exists)..."
conda env remove -n ctrlg -y 2>/dev/null || true

echo "Creating fresh Python 3.10 environment..."
conda create -n ctrlg python=3.10 -y
conda activate ctrlg

# 3. Install uv for high-speed package management
echo "Installing pip and uv..."
python -m pip install --upgrade pip
pip install uv

# 4. Install PyTorch + CUDA 12.1 (Optimized for H100)
# Using --index-url ensures we get the GPU binaries directly
echo "Installing PyTorch via uv..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install Faiss and other dependencies
# faiss-gpu-cu12 is the pip-compatible version for CUDA 12
echo "Installing Faiss and Transformers..."
uv pip install faiss-gpu-cu12 transformers==4.41.2 huggingface_hub==0.23.4 \
               sentencepiece protobuf notebook ipywidgets jupyterlab ipykernel

# 6. Setup the Ctrl-G Repository
echo "Setting up Ctrl-G repository..."
mkdir -p ~/CS5340/projects
cd ~/CS5340/projects

if [ ! -d Ctrl-G ]; then
  git clone https://github.com/joshuacnf/Ctrl-G
fi

cd Ctrl-G
git remote remove origin 2>/dev/null || true

# 7. Install Ctrl-G in editable mode
uv pip install -e .

# 8. Register Jupyter kernel for SOC Cluster JupyterHub (if used)
python -m ipykernel install --user --name ctrlg --display-name "Python (ctrlg)"

# 9. Final Sanity Check
echo "------------------------------------------"
echo "RUNNING SANITY CHECKS:"
python - <<'PY'
import torch
import faiss
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"Faiss GPU count: {faiss.get_num_gpus()}")
PY
echo "------------------------------------------"
echo "Setup finished successfully."