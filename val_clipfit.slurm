#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --account=su008-acw799

source ../env/torch24_env/bin/activate
export CLIPORT_ROOT=~/cliport
export PYTHONPATH=$PYTHONPATH:~/cliport

module purge      
module load CUDA/12.4.0
module load GCCcore/12.2.0 Python/3.10.8

srun python cliport/eval0.py eval_task=packing-boxes-pairs-seen-colors \
                       agent=clipfit \
                       mode=val \
                       n_demos=50 \
                       train_demos=10 \
                       checkpoint_type=val_missing \
                       exp_folder=exps_clipfit \
