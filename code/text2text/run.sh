#!/bin/bash

export TOKENIZERS_PARALLELISM=true

for j in  'amh' #'eng' 'fra' 'hau' 'ibo' 'lin' 'pcm' 'run' 'swa' 'yor' 
do
 
    for i in "masakhane/afri-byt5-base" #"castorini/afriteva_small" #"castorini/afriteva_base" "castorini/afriteva_large" #"masakhane/afri-mt5-base" #"masakhane/afri-byt5-base"
    do

        train_data_path="../../data/${j}/train.tsv"
        eval_data_path="../../data/${j}/dev.tsv"
        test_data_path="../../data/${j}/test.tsv"

        model_name_or_path=$i
        tokenizer_name_or_path=$i
        output_dir="output_"${i}-${j}
        lang=${j}

        max_seq_length="128"
        learning_rate="3e-4"
        train_batch_size="4"
        eval_batch_size="4"
        num_train_epochs="5"
        gradient_accumulation_steps="4"
        class_dir=../../data/${j}
        data_column="headline"
        target_column="category"
        prompt="classify: "

        python classification_trainer.py --train_data_path=$train_data_path \
                --eval_data_path=$eval_data_path \
                --test_data_path=$test_data_path \
                --model_name_or_path=$model_name_or_path \
                --tokenizer_name_or_path=$tokenizer_name_or_path \
                --output_dir=$output_dir \
                --max_seq_length=$max_seq_length \
                --train_batch_size=$train_batch_size \
                --eval_batch_size=$eval_batch_size \
                --num_train_epochs=$num_train_epochs \
                --gradient_accumulation_steps=$gradient_accumulation_steps \
                --class_dir=$class_dir \
                --target_column=$target_column \
                --data_column=$data_column \
                --prompt=$prompt \
                --learning_rate="3e-4" \
                --weight_decay="0.0" \
                --adam_epsilon="1e-8" \
                --warmup_steps="0" \
                --n_gpu="1" \
                --fp_16="false" \
                --max_grad_norm="1.0" \
                --opt_level="O1" \
                --seed="42" \
                --lang=$lang
                
                


            

    done
done