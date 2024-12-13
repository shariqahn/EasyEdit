#!/bin/bash

# Set up correct environment
# module load anaconda/2023b
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda deactivate 
conda activate easy

editing_method="SERAC"
# experiment="dummy"
# editing_method="ROME"
experiment="baseline"
data="../data/portability/One_Hop/zsre_mend_eval_portability_gpt4.json"
# "../data/dummy/zsre_mend_eval_portability_gpt4.json"
python -u run_zsre_llama2.py --editing_method "$editing_method" --hparams_dir "../hparams/${editing_method}/llama-7b.yaml" --data_file $data --metrics_save_dir "../outputs/${editing_method}_${experiment}/"

conda deactivate