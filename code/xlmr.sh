for sample in 5 10 20 50
do
  export MAX_LENGTH=256
  export BATCH_SIZE=16
  export NUM_EPOCHS=10
  export SAVE_STEPS=500000
  export BERT_MODEL=sentence-transformers/LaBSE

  for LANG in amh eng fra hau ibo lin orm pcm run sna swa yor
  do
    for j in 1 2 3 4 5
    do
      export SEED=$j
      export OUTPUT_FILE=test_result_${sample}_${LANG}_$j
      export OUTPUT_PREDICTION=test_predictions_${sample}_${LANG}_$j
      export DATA_DIR=../few_shot_data/${sample}_samples/${LANG}
      export OUTPUT_DIR=${sample}_${LANG}_labse

      CUDA_VISIBLE_DEVICES=0 python3 train_textclass.py --data_dir $DATA_DIR \
      --model_type xlmroberta \
      --model_name_or_path $BERT_MODEL \
      --output_dir $OUTPUT_DIR \
      --output_result $OUTPUT_FILE \
      --output_prediction_file $OUTPUT_PREDICTION \
      --max_seq_length  $MAX_LENGTH \
      --num_train_epochs $NUM_EPOCHS \
      --learning_rate 2e-5 \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --per_gpu_eval_batch_size $BATCH_SIZE \
      --save_steps $SAVE_STEPS \
      --seed $SEED \
      --gradient_accumulation_steps 2 \
      --do_train \
      --do_predict \
      --overwrite_output_dir
      done
    done
  done
