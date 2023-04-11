#!/bin/bash

export CUDA_AVAILABLE_DEVICES=0,1

# Train Masked Language Model

experiment_name=afriberta-distil


# Evaluate on Named Entity Recognition

#ner_model_path="${experiment_name}_ner_model"
LLM_path=$PWD/distillation/distil-4-4-gen
tokenizer_path=castorini/afriberta_large # specify tokenizer path or huggingface name

# mkdir $PWD/$ner_model_path
# mkdir $PWD/ner_results
output_dir=$PWD/General_AH4_HL4
mkdir $output_dir

# copy pretrained model from original folder to ner_model_path
cp $LLM_path/pytorch_model.bin $output_dir/pytorch_model.bin
cp $LLM_path/config.json $output_dir/config.json

MAX_LENGTH=164
MODEL_PATH=$output_dir
BATCH_SIZE=16
NUM_EPOCHS=50
SAVE_STEPS=1000
TOK_PATH=$tokenizer_path
declare -a arr=("hau" "ibo" "kin" "lug" "luo" "pcm" "swa" "wol" "yor")

for SEED in 1 3  5
do
    # output_dir=ner_results/"${experiment_name}_${SEED}"
    # mkdir $PWD/$output_dir

    for i in "${arr[@]}"
    do
        OUTPUT_DIR=$output_dir/"${i}_${SEED}"
        DATA_DIR=ner_data/"$i"
        python ner_scripts/train_ner.py --data_dir $DATA_DIR \
        --model_type nil \
        --model_name_or_path $MODEL_PATH \
        --tokenizer_path $TOK_PATH \
        --output_dir $OUTPUT_DIR \
        --max_seq_length $MAX_LENGTH \
        --num_train_epochs $NUM_EPOCHS \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --per_gpu_eval_batch_size $BATCH_SIZE \
        --save_steps $SAVE_STEPS \
        --seed $SEED \
        --do_train \
        --do_eval \
        --do_predict \
        --csv_file general-AH4-HL4_results.csv \
        --lang_seed "${i}_${SEED}" \
        --wandb_entity_name compression_on_afriberta\
        --wandb_project_name general-distil-ner

    done
done
