# #!/bin/bash

# python -u memit.py

# # ~8m
# apt update -y
# apt install rsync -y
# rsync -avz -e ssh /workspace/outputs/ shossain@txe1-login.mit.edu:/home/gridsan/shossain/EasyEdit/outputs/ -y

#!/bin/bash

# 1. Run the Python script in the background (using nohup)
echo "Starting the Python script..."
nohup python3 -u memit.py > output.log 2>&1 &  # Running the script in the background
PID=$!  # Capture the PID of the running Python script

echo "Python script is running with PID: $PID"

# 2. Wait for the script to finish (you can modify this to wait longer if needed)
echo "Waiting for the Python script to finish..."
wait $PID

echo "Python script finished!"

# 3. Sync the output file to your local machine using rsync
# Replace <your_local_path> with the actual path on your local machine
echo "Syncing output file to local laptop..."
# apt update -y
# apt install rsync -y
rsync -avz -e ssh /workspace/outputs/ shossain@txe1-login.mit.edu:/home/gridsan/shossain/EasyEdit/outputs/ -y
echo "Sync complete!"
