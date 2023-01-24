#!/bin/bash
#SBATCH --job-name=masakhanews
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/YOSM/slurm/slurmerror_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/YOSM/slurm/slurmoutput_%j.txt


cd /home/mila/c/chris.emezue/YOSM
module load python/3
module load cuda/11.0/cudnn/8.0
source /home/mila/c/chris.emezue/scratch/bam_env/bin/activate
#export PYTHONPATH=$PYTHONPATH:/home/mila/c/chris.emezue/alpha-team-mt-competition
#export MKL_SERVICE_FORCE_INTEL=1
#export LD_LIBRARY_PATH=/home/mila/c/chris.emezue/scratch/nemo2/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH



export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased
export OUTPUT_DIR=/home/mila/c/chris.emezue/scratch/masakhanews/yo_mbert/$2/$3/$1
export OUTPUT_FILE=test_result_$2_$3_$1
export OUTPUT_PREDICTION=test_predictions_$2_$3_$1
export BATCH_SIZE=32
export NUM_EPOCHS=20
export SAVE_STEPS=5000
export SEED=$1


python3 train_textclass.py --data_dir /home/mila/c/chris.emezue/scratch/masakhanews/masakhane-news/data/$2/ \
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
--overwrite_output_dir \
--labels /home/mila/c/chris.emezue/scratch/masakhanews/masakhane-news/data/$2/labels.txt \
--headline_use $3
