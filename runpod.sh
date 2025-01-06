#!/bin/bash

# git clone https://github.com/shariqahn/EasyEdit.git
# cd EasyEdit
echo "set up ssh!"
ssh-keyscan txe1-login.mit.edu >> ~/.ssh/known_hosts
ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N ""
cat ~/.ssh/id_rsa.pub
apt update -y
apt install rsync -y

echo "pwd:"
pwd
git checkout memit
pip install --no-cache-dir -r requirements.txt

mkdir /workspace/hf
echo "workspace:"
ls /workspace
export HF_TOKEN=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

cd examples
echo "pwd:"
# 1. Run the Python script in the background (using nohup)
echo "Starting the Python script..."
editing_method="LoRA"
experiment="baseline"
data="../data/notebook/zsre_mend_eval_portability_gpt4.json"
nohup python3 -u run_lora.py \
  --editing_method "$editing_method" \
  --hparams_dir "../hparams/${editing_method}/notebook.yaml" \
  --data_file "$data" \
  --metrics_save_dir "/workspace/outputs/${editing_method}_${experiment}/" \
  > output.log 2>&1 &

PID=$!  # Capture the PID of the running Python script

echo "Python script is running with PID: $PID"

# 2. Wait for the script to finish (you can modify this to wait longer if needed)
echo "Waiting for the Python script to finish..."
wait $PID

echo "Python script finished!"

# 3. Sync the output file to your local machine using rsync
echo "Syncing output file to local laptop..."
rsync -avz -e ssh /workspace/outputs/ shossain@txe1-login.mit.edu:/home/gridsan/shossain/EasyEdit/outputs/ -y
echo "Sync complete!"
