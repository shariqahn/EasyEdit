#!/bin/bash

# Set up correct environment
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

# editing_method="MEMIT"
editing_method="IKE"
experiment="dummy_tofu"
data="../data/tofu_test_dummy_zsre.json"
# data="../data/portability/One_Hop/zsre_mend_eval_portability_gpt4.json"
# data="../data/counterfact/counterfact-edit.json"
python -u run_zsre_llama2.py --editing_method "$editing_method" --hparams_dir "../hparams/${editing_method}/llama-7b.yaml" --data_file $data --metrics_save_dir "../outputs/${editing_method}_${experiment}/"
# python -u run_counterfact_gpt.py --editing_method "$editing_method" --hparams_dir "../hparams/${editing_method}/gpt-j-6B.yaml" --data_file $data --metrics_save_dir "../outputs/${editing_method}_${experiment}/"

conda deactivate