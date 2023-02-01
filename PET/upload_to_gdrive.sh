#!/bin/bash
#SBATCH --job-name=upload
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --error=/home/mila/c/chris.emezue/pet-masakhane-news/slurmerror_upload_%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/pet-masakhane-news/slurmoutput_upload_%j.txt


cd /home/mila/c/chris.emezue/scratch/pet-masakhane-results2/pet-masakhane
module load python/3
module load cuda/11.0/cudnn/8.0
source /home/mila/c/chris.emezue/scratch/pet-env/bin/activate

#export PYTHONPATH=$PYTHONPATH:/home/mila/c/chris.emezue/alpha-team-mt-competition
#export MKL_SERVICE_FORCE_INTEL=1
#export LD_LIBRARY_PATH=/home/mila/c/chris.emezue/scratch/nemo2/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

#gdrive upload -r pet-masakhane

#for file in `find ./ -type f -name "*.txt"`
gdrive upload -r -p 1jtCdy0-TPDRkA_9BXys5JMCR0C9-SRGB results_pet/
