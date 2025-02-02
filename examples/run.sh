#!/bin/bash

# STEPS
# redownload model
# update params below
# make sure correct model in hparams
    # if SERAC, update archive
# make sure portability, locality, sequential_edit are used appropriately in run_zsre_llama2.py
# LLsub run.sh -s 40 -g volta:2

# Set up correct environment
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

# editing_method="MEMIT"
editing_method="ROME"
experiment="bias"
# data="../data/tofu_test_dummy_zsre.json"
# data="../data/portability/One_Hop/zsre_mend_eval_portability_gpt4.json"
# data="../data/tofu_test_zsre.json"
# data="../data/counterfact/counterfact-edit.json"
data="../data/bias.json"
# python -u run_zsre_llama2.py --editing_method "$editing_method" --hparams_dir "../hparams/${editing_method}/llama-7b.yaml" --data_file $data --metrics_save_dir "../outputs/${editing_method}_${experiment}"
python -u bias.py --editing_method "$editing_method" --hparams_dir "../hparams/${editing_method}/llama-7b.yaml" --data_file $data --metrics_save_dir "../outputs/${editing_method}_${experiment}"
# python -u run_counterfact_gpt.py --editing_method "$editing_method" --hparams_dir "../hparams/${editing_method}/gpt-j-6B.yaml" --data_file $data --metrics_save_dir "../outputs/${editing_method}_${experiment}"

conda deactivate