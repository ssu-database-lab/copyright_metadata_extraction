#!/usr/bin/env bash
# Fix for adapters/torch compatibility issue by disabling distributed tensor

export TORCH_CUDA_ARCH_LIST=""
export CUDA_LAUNCH_BLOCKING=1

cd /mnt/c/Users/peppermint/Desktop/copyright_metadata_extraction/metadata

# Activate venv
source .venv/bin/activate

# Run Python with monkey-patch to fix the import issue
python3 << 'EOF'
import sys
import torch

# Disable distributed tensor to avoid import issues  
torch.distributed.is_available = lambda: False

# Now run the actual training
from module import api

api.ner_train(
    model_name="bert-base-multilingual-cased",
    model_path=None,
    adapter_dir="models/ner/adapters",
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    train_data_path="configs/training",
    train_ratio=0.8,
    random_seed=42
)
EOF
