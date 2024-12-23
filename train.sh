#!/bin/bash

# Set up correct environment
# module load anaconda/2023b
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

# STEPS
# redownload model
# update data in script
# make sure correct model in hparams
# LLsub train.sh -s 40 -g volta:2

python -u serac_train.py

conda deactivate