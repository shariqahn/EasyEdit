#!/bin/bash

# Set up correct environment
# module load anaconda/2023b
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

python -u run_zsre_llama2.py --editing_method "ROME" --hparams_dir "../hparams/ROME/llama-7b.yaml" --data_dir "../data/portability/One_Hop"

conda deactivate