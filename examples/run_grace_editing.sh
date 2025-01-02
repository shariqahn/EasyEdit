#!/bin/bash

# Set up correct environment
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

# STEPS
# redownload model
# update params below
# make sure correct model in hparams

# experiment="avoidant"
experiment="incorrect"
# data="../data/tofu_locality.json"
data="../data/avoidant.json"

CUDA_VISIBLE_DEVICES=0 python -u run_grace_editing.py \
  --editing_method=GRACE \
  --hparams_dir=../hparams/GRACE/llama-7b.yaml \
  --data_file=$data \
  --data_type=ZsRE \
  --sequential_edit \
  --experiment=$experiment \
  --output_dir=../outputs/GRACE_${experiment}
  
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