export MAX_LENGTH=164
export OUTPUT_FILE=test_result
export OUTPUT_PREDICTION=test_predictions
export BATCH_SIZE=64
export NUM_EPOCHS=10
export SAVE_STEPS=500000
export SEED=1
export LANG=pcm
export DATA_DIR=../data/${LANG}
	

export BERT_MODEL=Davlan/afro-xlmr-base
export OUTPUT_DIR=${LANG}_afroxlmrbase

CUDA_VISIBLE_DEVICES=3 python3 train_textclass.py --data_dir $DATA_DIR \
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
--labels ../data/${LANG}/labels.txt
--gradient_accumulation_steps 2 \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir