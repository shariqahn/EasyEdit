from huggingface_hub import snapshot_download
import getpass
import transformers

# snapshot_download(repo_id=config.model.name, cache_dir=cache_dir)
# snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2')

from transformers import AutoTokenizer, AutoModelForCausalLM




if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"  # Hugging Face model path for LLaMA 2 7B
    cache_dir="/state/partition1/user/" + getpass.getuser() + "/hug"
    print('downloading llama')
    snapshot_dir = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    print(f'snapshot: {snapshot_dir}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=snapshot_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=snapshot_dir)

    # Example usage
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))