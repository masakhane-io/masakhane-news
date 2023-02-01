#!/bin/bash
#SBATCH --job-name=pet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/pet-masakhane-news/slurm/slurmerror_pet_install_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/pet-masakhane-news/slurm/slurmoutput_pet_install_%j.txt


cd /home/mila/c/chris.emezue/pet-masakhane-news
module load python/3
module load cuda/11.0/cudnn/8.0
source /home/mila/c/chris.emezue/scratch/pet-env/bin/activate

pip install --upgrade pip
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt