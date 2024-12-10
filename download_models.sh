#!/bin/bash

# Set up correct environment
# module load anaconda/2023b
# source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
# conda deactivate 
# conda activate easy

# export HYDRA_FULL_ERROR=1
# This is where model is downloaded
export HF_HOME=/state/partition1/user/$USER/hug
mkdir -p $HF_HOME
HF_LOCAL_DIR=$HOME/EasyEdit/scr
mkdir -p $HF_LOCAL_DIR

# Remove existing models so that they will be replaced with fresh ones
# rm -r $HF_LOCAL_DIR/*
rm -r $HF_LOCAL_DIR/models--locuslab--tofu_ft_llama2-7b/
echo "Existing models removed. Here's what local looks like:"
ls $HF_LOCAL_DIR

# Token so can access restricted HuggingFace models
export HF_TOKEN=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

echo "Dirs created:"
ls /state/partition1/user/$USER
echo "downloading"
python -u download.py
# echo "downloading rest:"
# cd examples
# python -u run_zsre_llama2.py --editing_method "ROME" --hparams_dir "../hparams/ROME/llama-7b-download.yaml" --data_dir "../data/portability/One_Hop"

# Copy the model from HF_HOME into HF_LOCAL_DIR
echo "Model collected. Here is what home looks like:"
ls $HF_HOME
rsync -a --ignore-existing ${HF_HOME}/ $HF_LOCAL_DIR
echo "Model copied. Here is what local looks like:"
ls $HF_LOCAL_DIR
rm -rf $HF_HOME
echo "Home cleared. Here is what home looks like:"
ls /state/partition1/user/$USER