#!/bin/bash

# Read all generated sweep IDs into an array
mapfile -t SWEEP_IDS < sweep_ids.txt

# Available GPU IDs
GPU_IDS=(0 1)

# Number of concurrent agents per GPU
THREADS_PER_GPU=4

echo "Detected ${#SWEEP_IDS[@]} sweep jobs to run."

# Launch agents and distribute jobs across GPUs
index=0
for GPU_ID in "${GPU_IDS[@]}"; do
    for ((i=0; i<THREADS_PER_GPU; i++)); do
        if [ $index -lt ${#SWEEP_IDS[@]} ]; then
            SID=${SWEEP_IDS[$index]}
            echo "GPU $GPU_ID is assigned to sweep: $SID"

            # Limit the number of runs per agent; after completion, it can take new tasks if needed
            CUDA_VISIBLE_DEVICES=$GPU_ID nohup wandb agent "$SID" --count 20 \
                > "log_${GPU_ID}_${index}.txt" 2>&1 &

            let index++
            sleep 2
        fi
    done
done

echo "All agents have been launched. Monitor progress on the WandB dashboard."