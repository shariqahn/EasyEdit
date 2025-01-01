#!/bin/bash

# Set up correct environment
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

# Load the .env file
if [ -f ../.env ]; then
    export $(grep -v '^#' ../.env | xargs)
else
    echo ".env file not found!"
    exit 1
fi

# export OPENAI_API_KEY=$OPENAI_KEY
export HF_HOME=/state/partition1/user/$USER/hug
# Token so can access restricted HuggingFace models
export HF_TOKEN=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
python -u avoid_targets.py