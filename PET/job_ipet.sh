#!/bin/bash
#SBATCH --job-name=ipet
#SBATCH --gres=gpu:24GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/pet-masakhane-news/slurm/slurmerror_ipet_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/pet-masakhane-news/slurm/slurmoutput_ipet_%j.txt


cd /home/mila/c/chris.emezue/pet-masakhane-news
module load python/3
module load cuda/11.0/cudnn/8.0
source /home/mila/c/chris.emezue/scratch/pet-env/bin/activate



python3 cli.py --method ipet --pattern_ids 0 1 2 3 4 \
--data_dir /home/mila/c/chris.emezue/pet-masakhane-news/data-fsl/$1/$2sample/ \
--model_type xlm-roberta \
--model_name_or_path Davlan/afro-xlmr-large \
--task_name topic-classification \
--output_dir /home/mila/c/chris.emezue/scratch/pet-masakhane/results_ipet/$1/model-ipet-$2-Davlan-afro-xlmr-large/ \
--do_train \
--do_eval \
--sc_repetitions 1 \
--overwrite_output_dir \
--verbalizer_file $1