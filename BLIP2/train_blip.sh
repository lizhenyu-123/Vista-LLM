#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

PYTHON_EXECUTABLE="python"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

echo "Working directory has been changed to project root: $(pwd)"
echo "Run started at: $(date)"

$PYTHON_EXECUTABLE -m BLIP2.train_blip \
    --root_dir "/path/to/dataset/VG" \
    --output_dir "./output_blip2_keytoken" \
    --model_name "/path/to/blip2-itm-vit-g" \
    # --resume_from_checkpoint "/path/to/checkpoint" \
