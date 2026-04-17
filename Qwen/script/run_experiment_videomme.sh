#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export USE_AGGREGATION=True 
export TAU_SIMILARITY_THRESHOLD=0.80
export RETENTION_RATIO=0.25
export R_MIN_RATIO=0.10

PYTHON_EXECUTABLE="python"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."
echo "Working directory has been changed to project root: $(pwd)"

TASKS=(
    "short"
    "medium"
    "long"
)

DOMAINS=(
    0.85
)

echo "--- Starting VideoMME Inference Run ---"
echo "Run started at: $(date)"
echo "=================================================="

for domain in "${DOMAINS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo -e "\n=================================================="
        echo "STARTING TASK: $task | DOMAIN RATE: $domain"
        echo "=================================================="

        $PYTHON_EXECUTABLE -m script.run \
            --dataset "videomme" \
            --model-path "path/to/qwen2.5-vl" \
            --train-path "path/to/trained_checkpoint" \
            --output-dir "./results/results_videomme" \
            --videomme-ann-file "/path/to/test-00000-of-00001.parquet" \
            --frames-root "path/to/videomme_frames" \
            --video-root "path/to/videomme_videos" \
            --task-type "$task" \
            --domain-rate $domain \
            --use-preextracted-frames \
            --use-way 

        if [ $? -eq 0 ]; then
            echo "Task '$task' completed successfully."
        else
            echo "ERROR: Script failed for task '$task'."
        fi
    done
done

echo -e "\n=================================================="
echo "All tasks have been processed!"
echo "=================================================="