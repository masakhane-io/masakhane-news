for j in 1 2 3 4 5
do
	export MAX_LENGTH=128
	export BERT_MODEL=Davlan/bert-base-multilingual-cased-finetuned-yoruba
	export OUTPUT_DIR=yo_laft_bert
	export OUTPUT_FILE=test_result$j
	export OUTPUT_PREDICTION=test_predictions
	export BATCH_SIZE=32
	export NUM_EPOCHS=20
	export SAVE_STEPS=5000
	export SEED=$j

	CUDA_VISIBLE_DEVICES=0 python3 train_textclass.py --data_dir data/yosm/yosm/ \
	--model_type bert \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--output_result $OUTPUT_FILE \
	--output_prediction_file $OUTPUT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_predict \
	--overwrite_output_dir
done
