#!/bin/bash
# Install local LLM for deeplogbot unsupervised classification
# This script installs llama-cpp-python and downloads a recommended model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models/llm"

echo "========================================"
echo "DeepLogBot Local LLM Setup"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Install llama-cpp-python
echo ""
echo "Step 1: Installing llama-cpp-python..."
echo "----------------------------------------"

# Check if running on Apple Silicon for Metal acceleration
if [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    echo "Detected Apple Silicon - enabling Metal acceleration"
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade
elif command -v nvidia-smi &> /dev/null; then
    echo "Detected NVIDIA GPU - enabling CUDA acceleration"
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade
else
    echo "No GPU detected - installing CPU-only version"
    pip install llama-cpp-python --upgrade
fi

# Create models directory
echo ""
echo "Step 2: Creating models directory..."
echo "----------------------------------------"
mkdir -p "$MODELS_DIR"

# Download recommended model (Mistral 7B Instruct Q4_K_M - good balance of quality/speed)
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_FILE="$MODELS_DIR/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

echo ""
echo "Step 3: Downloading Mistral 7B Instruct model..."
echo "----------------------------------------"
echo "Model: Mistral 7B Instruct v0.2 (Q4_K_M quantization)"
echo "Size: ~4.4 GB"
echo "Destination: $MODEL_FILE"
echo ""

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_FILE"
    echo "Skipping download. Delete the file to re-download."
else
    echo "Downloading... (this may take a few minutes)"

    # Use curl or wget
    if command -v curl &> /dev/null; then
        curl -L -o "$MODEL_FILE" "$MODEL_URL" --progress-bar
    elif command -v wget &> /dev/null; then
        wget -O "$MODEL_FILE" "$MODEL_URL" --show-progress
    else
        echo "ERROR: Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    echo "Download complete!"
fi

# Create example config
CONFIG_DIR="$PROJECT_ROOT/config/llm_configs"
mkdir -p "$CONFIG_DIR"

LOCAL_CONFIG="$CONFIG_DIR/local_llm.json"
echo ""
echo "Step 4: Creating configuration file..."
echo "----------------------------------------"

cat > "$LOCAL_CONFIG" << EOF
{
    "provider": "local",
    "model_path": "$MODEL_FILE",
    "context_length": 4096,
    "temperature": 0.1,
    "max_tokens": 500,
    "n_gpu_layers": -1
}
EOF

echo "Created config at: $LOCAL_CONFIG"

# Verify installation
echo ""
echo "Step 5: Verifying installation..."
echo "----------------------------------------"

python3 << 'PYEOF'
try:
    from llama_cpp import Llama
    print("llama-cpp-python installed successfully!")
except ImportError as e:
    print(f"ERROR: Failed to import llama-cpp-python: {e}")
    exit(1)
PYEOF

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To use the local LLM for unsupervised classification:"
echo ""
echo "  python -m logghostbuster --classification-method unsupervised-llm \\"
echo "    --llm-config $LOCAL_CONFIG \\"
echo "    --input your_data.parquet"
echo ""
echo "Alternative models (smaller/faster):"
echo "  - Phi-2 (2.7B): https://huggingface.co/TheBloke/phi-2-GGUF"
echo "  - TinyLlama (1.1B): https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
echo ""
echo "For GPU acceleration, the model will automatically use:"
echo "  - Apple Silicon: Metal"
echo "  - NVIDIA: CUDA (if available during install)"
echo ""
