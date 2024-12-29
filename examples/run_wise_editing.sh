#!/bin/bash

# Set up correct environment
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

# STEPS
# redownload model
# update params below
# make sure correct model in hparams
# make sure portability, locality, sequential_edit are used appropriately in run_tofu.py

# editing_method="SERAC"
experiment="incorrect"
editing_method="ROME"
# experiment="dummy"
data="./data/tofu_locality.json"
CUDA_VISIBLE_DEVICES=0 python run_wise_editing.py \
  --editing_method=GRACE \
  --hparams_dir=../hparams/GRACE/llama-7b.yaml \
  --data_dir=../data/wise \
  --ds_size=10 \
  --data_type=ZsRE \
  --sequential_edit
# python -u run_tofu.py --editing_method "$editing_method" --hparams_dir "./hparams/${editing_method}/llama-7b.yaml" --data_file $data --experiment $experiment --metrics_save_dir "./outputs/${editing_method}_${experiment}_lr_1_clamp_2/"

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