#!/bin/bash

# Set up correct environment
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda activate easy

# STEPS
# redownload model
# update params below
# make sure correct model in hparams
# make sure portability, locality, sequential_edit are used appropriately in run_tofu.py
# put correct yaml; model reltive path

# editing_method="SERAC"
editing_method="ROME"
experiment="avoidant"
data="./data/avoidant.json"
python -u run_tofu.py --editing_method "$editing_method" --hparams_dir "./hparams/${editing_method}/llama-7b.yaml" --data_file $data --experiment $experiment --metrics_save_dir "./outputs/${editing_method}_${experiment}_lr_1/"

conda deactivate