#!/bin/bash

# Set up correct environment
# module load anaconda/2023b
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

# editing_method="SERAC"
experiment="dummy"
editing_method="IKE"
# experiment="incorrect"
# data="./data/dummy/zsre_mend_eval_portability_gpt4.json"
data="./data/tofu_subject.json"
python -u run_tofu.py --editing_method "$editing_method" --hparams_dir "./hparams/${editing_method}/llama-7b.yaml" --data_file $data --experiment $experiment --metrics_save_dir "./outputs/${editing_method}_${experiment}_sequential/"

conda deactivate