from huggingface_hub import snapshot_download
import getpass
import pdb
import os 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    # for model_name in ["meta-llama/Llama-2-7b-hf", 'Cheng98/llama-160m', "distilbert/distilbert-base-cased"]:
        # "JackFram/llama-160m"
    # for model_name in ["architext/gptj-162M", "distilbert/distilbert-base-cased", 'EleutherAI/gpt-j-6b']:
    # ["locuslab/tofu_ft_llama2-7b", "sentence-transformers/all-MiniLM-L6-v2"]:
    for model_name in ["meta-llama/Llama-2-7b-hf"]:
    # for model_name in ["locuslab/tofu_ft_llama2-7b"]:
    
    # 
    # model_name = "openai-community/gpt2-xl"
        cache_dir="/state/partition1/user/" + getpass.getuser() + "/hug"
        print(f'downloading {model_name}\n')
        snapshot_dir = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
        print(f'snapshot: {snapshot_dir}\n')
    
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
