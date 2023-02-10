for SRC_LANG in hau swa
do
  export MAX_LENGTH=164
  export BATCH_SIZE=16
  export NUM_EPOCHS=10
  export SAVE_STEPS=500000
  export BERT_MODEL=Davlan/afro-xlmr-base

  for LANG in amh eng fra hau ibo lin lug orm pcm run sna swa yor
  do
    for j in 1 2 3 4 5
    do
      export SEED=$j
      export OUTPUT_FILE=test_result_${SRC_LANG}_${LANG}
      export OUTPUT_PREDICTION=test_predictions_${SRC_LANG}_${LANG}
      export DATA_DIR=../data/${LANG}
      export OUTPUT_DIR=${SRC_LANG}_afroxlmrbase${SEED}

      CUDA_VISIBLE_DEVICES=2 python3 train_textclass.py --data_dir $DATA_DIR \
      --model_type xlmroberta \
      --model_name_or_path $BERT_MODEL \
      --output_dir $OUTPUT_DIR \
      --output_result $OUTPUT_FILE \
      --output_prediction_file $OUTPUT_PREDICTION \
      --max_seq_length  $MAX_LENGTH \
      --num_train_epochs $NUM_EPOCHS \
      --learning_rate 5e-5 \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --per_gpu_eval_batch_size $BATCH_SIZE \
      --save_steps $SAVE_STEPS \
      --seed $SEED \
      --gradient_accumulation_steps 2 \
      --do_predict \
      --overwrite_output_dir
      done
    done
  done