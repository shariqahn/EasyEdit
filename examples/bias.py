import os.path
import sys
sys.path.append('..')
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

import argparse

import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    # parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    # elif args.editing_method == 'IKE':
    #     editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)

    # test_data = json.load(open(os.path.join(args.data_dir, 'zsre_mend_eval_portability_gpt4.json'), 'r', encoding='utf-8'))
    test_data = json.load(open(args.data_file, 'r', encoding='utf-8'))
    counterfact = "../data/counterfact/counterfact-edit.json"
    locality_data = json.load(open(counterfact, 'r', encoding='utf-8'))

    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)

    prompts = [test_data_['prompt'] for test_data_ in test_data]
    rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in test_data]
    target_new = [edit_data_['target_new'] for edit_data_ in test_data]
    locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in locality_data]
    locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in locality_data]
    # locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in test_data]
    # locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in test_data]
    # portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
    # portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]

    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts[:len(prompts)],
            'ground_truth': locality_ans[:len(prompts)]
        },
    }
    # portability_inputs = {
    #     'one_hop':{
    #         'prompt': portability_prompts,
    #         'ground_truth': portability_ans
    #     },
    # }
    subject = [edit_data_['subject'] for edit_data_ in test_data]

    # if args.editing_method == 'IKE':
    #     train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
    #     train_ds = ZsreDataset(train_data_path)
    #     sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    #     encode_ike_facts(sentence_model, train_ds, hparams)
    # else:
    train_ds = None

    sequential_edit = False
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        # portability_inputs=portability_inputs,
        # keep_original_weight=True
        keep_original_weight=False,
        sequential_edit=sequential_edit
    )

    os.makedirs(args.metrics_save_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results.json'), 'w'), indent=4)
    # snh so can load model later
    model_save_dir = os.path.join(args.metrics_save_dir, 'model')
    os.makedirs(model_save_dir, exist_ok=True)
    edited_model.save_pretrained(model_save_dir)
