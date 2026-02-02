#!/bin/bash

TASKS=("plot") # "("character" "plot")
MODELS=("llama-70b") # ("gpt-5.1" "gpt-5.2" "exaone-7b" "exaone-32b" "llama-7b" "llama-70b" "qwen-8b" "qwen-14b" "gpt-oss-20b")
DATA_SEEDS=(42 43 44 45 46) #(42 43 44 45 46)
MODEL_SEEDS=(42 43) #(42 43)

INPUT_DIR="./data"
OUTPUT_DIR="./results"
SAMPLING_COUNT=10 # 10
TEMPERATURE=0.0

for task in "${TASKS[@]}"; do
    
    if [ "$task" == "character" ]; then
        INPUT_FILE="character_s3_full_final_len_67566.csv"
    elif [ "$task" == "plot" ]; then
        INPUT_FILE="plot_s3_full_final_len_67676.csv"
    else
        echo "Unknown task: $task"
        continue
    fi

    echo "Task: $task | Input File: $INPUT_FILE"

    for model in "${MODELS[@]}"; do
        for d_seed in "${DATA_SEEDS[@]}"; do
            for m_seed in "${MODEL_SEEDS[@]}"; do
                
                echo "----------------------------------------------------------------"
                echo ":: Running: Model=$model | DataSeed=$d_seed | ModelSeed=$m_seed::"
                echo "----------------------------------------------------------------"

                python code/main.py \
                    --input_path "$INPUT_DIR" \
                    --input_file "$INPUT_FILE" \
                    --output_path "$OUTPUT_DIR" \
                    --task_type "$task" \
                    --model_name "$model" \
                    --dataset_seed "$d_seed" \
                    --model_seed "$m_seed" \
                    --sampling_count "$SAMPLING_COUNT" \
                    --temperature "$TEMPERATURE" \
                    --max_token 64 \
                    --run_type "test"


                if [ $? -ne 0 ]; then
                    echo "Error occurred during execution."
                    exit 1 
                fi
                
                echo ""
            done
        done
    done
done

echo "All experiments completed."