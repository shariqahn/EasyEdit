from huggingface_hub import snapshot_download
import getpass
import pdb
import os 
from datasets import load_dataset

# snapshot_download(repo_id=config.model.name, cache_dir=cache_dir)
# snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2')

from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    model_name = "locuslab/tofu_ft_llama2-7b"
    # "meta-llama/Llama-2-7b-hf"  # Hugging Face model path for LLaMA 2 7B
    # model_name = "openai-community/gpt2-xl"
    cache_dir="/state/partition1/user/" + getpass.getuser() + "/hug"
    print(f'downloading {model_name}')
    snapshot_dir = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    print(f'snapshot: {snapshot_dir}')
    
    # download MEMIT data for easyeditor/models/rome/layer_stats.py
    # ds_name = 'wikipedia'
    # raw_ds = load_dataset(
    #         ds_name,
    #         dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name]
    #     )
    # # ['train']
    # save_path='wikipedia'
    # raw_ds.save_to_disk(os.path.join(cache_dir, save_path))

    # from datasets import load_from_disk
    # scr = '/home/gridsan/shossain/EasyEdit/scr'
    # ds_name='wikipedia'
    # raw_ds = load_from_disk(f'{scr}/{ds_name}')
