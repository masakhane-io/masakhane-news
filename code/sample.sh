#!/bin/bash
#SBATCH --job-name=masakhanews-experiments
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=96G
#SBATCH --time=168:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/b/bonaventure.dossou/masakhane-news/comErrorXLMRBase.txt
#SBATCH --output=/home/mila/b/bonaventure.dossou/masakhane-news/comOutputXLMRBase.txt

###########cluster information above this line
module load python/3.9 cuda/10.2/cudnn/7.6
source /home/mila/b/bonaventure.dossou/afrispeech/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:~/masakhane-news
export MAX_LENGTH=164
export BATCH_SIZE=64
export NUM_EPOCHS=10
export SAVE_STEPS=1500

for model in xlm-roberta-base xlm-roberta-large google/rembert microsoft/deberta-v3-base
do
	export BERT_MODEL=${model}
	export OUTPUT_DIR=/home/mila/b/bonaventure.dossou/masakhane-news/results/${LANG}_${model}
	for lang in amh eng fra hau ibo lin pcm run swa yor
	do
		export DATA_DIR=/home/mila/b/bonaventure.dossou/masakhane-news/data/${LANG}
		export LABELS_FILE=${DATA_DIR}/labels.txt
		export LANG=${lang}
		export OUTPUT_DIR=/home/mila/b/bonaventure.dossou/masakhane-news/results/${LANG}_${model}
		for seed in 1 2 3 4 5
		do
			for header_style in 0 1
			do

				export SEED=${seed}
				export OUTPUT_FILE=/home/mila/b/bonaventure.dossou/masakhane-news/results/test_result_${LANG}_${seed}
				export OUTPUT_PREDICTION=/home/mila/b/bonaventure.dossou/masakhane-news/results/test_predictions_${LANG}_${seed}

				CUDA_VISIBLE_DEVICES=2 python train_textclass.py --data_dir $DATA_DIR \
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
				--labels $LABELS_FILE \
				--save_total_limit 1 \
				--gradient_accumulation_steps 2 \
				--do_train \
				--do_eval \
				--do_predict \
				--overwrite_output_dir \
				--header header_style
			done
		done
	done
done