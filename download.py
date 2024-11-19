from huggingface_hub import snapshot_download
import getpass
import pdb
import torch

# snapshot_download(repo_id=config.model.name, cache_dir=cache_dir)
# snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2')

from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    # model_name = "meta-llama/Llama-2-7b-hf"  # Hugging Face model path for LLaMA 2 7B
    # meta-llama/Llama-2-7b-chat-hf
    model_name = "openai-community/gpt2-xl"
    cache_dir="/state/partition1/user/" + getpass.getuser() + "/hug"
    print(f'downloading {model_name}')
    snapshot_dir = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    print(f'snapshot: {snapshot_dir}')
    # pdb.set_trace()

    # model = AutoModelForCausalLM.from_pretrained(snapshot_dir, device_map=None, local_files_only=True, torch_dtype=torch.float32)
    # tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)

    # # # Define the model name and path
    # # model_name = '/home/gridsan/shossain/EasyEdit/scr/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'

    # # # Load the model and tokenizer using the transformers library
    # # model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype=torch.float32, local_files_only=True)
    # # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    # # pdb.set_trace()

    # # Example usage
    # input_text = "Where is the eiffel tower?"
    # inputs = tokenizer(input_text, return_tensors="pt")
    # inputs = inputs.to('cuda')
    # outputs = model.generate(inputs['input_ids'], max_length=50)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
