#!/bin/bash

# Set up correct environment
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
# conda deactivate 
conda activate easy
which python
conda info --envs

# STEPS
# redownload model
# update params below
  # make sure cuda devices are correct
# make sure correct model in hparams
# TODO DONT FORGET TO CHANGE STUFF BACK HPARAMS, PDB, CUDA ETC

experiment="dummy"
# experiment="incorrect"
# data="../data/tofu_locality.json"
data="../data/avoidant.json"
# data="../data/wise"
editing_method="WISE"

CUDA_VISIBLE_DEVICES=0 python -u run_grace_editing.py \
  --editing_method=$editing_method \
  --hparams_dir=../hparams/${editing_method}/llama-7b.yaml \
  --data_file=$data \
  --data_type=ZsRE \
  --sequential_edit \
  --experiment=$experiment \
  --output_dir=../outputs/${editing_method}_${experiment}
  
conda deactivate

# CUDA_VISIBLE_DEVICES=0 python run_wise_editing.py \
#   --editing_method=WISE \
#   --hparams_dir=../hparams/WISE/llama-3-8b.yaml \
#   --data_dir=../data/wise \
#   --ds_size=10 \
#   --data_type=ZsRE \
#   --sequential_edit


#CUDA_VISIBLE_DEVICES=0 python run_wise_editing.py \
#   --editing_method=WISE \
#   --hparams_dir=../hparams/WISE/llama-7b \
#   --data_dir=../data/wise \
#   --ds_size=3 \
#   --data_type=temporal \
#   --sequential_edit


#python run_wise_editing.py \
#  --editing_method=WISE \
#  --hparams_dir=../hparams/WISE/llama-7b \
#  --data_dir=../data/wise \
#  --ds_size=3 \
#  --data_type=hallucination
##  --sequential_edit