import os.path
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

# snh
from datasets import load_from_disk
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--experiment', required=True, type=str)

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

    test_data = json.load(open(args.data_file, 'r', encoding='utf-8'))

    if args.ds_size is not None:
        test_data = random.sample(test_data, args.ds_size)

    prompts = [test_data_['question'] for test_data_ in test_data]
    rephrase_prompts = [edit_data_['paraphrased_question'] for edit_data_ in test_data]
    if args.experiment == 'incorrect':
        target_new = [edit_data_['perturbed_answer'][0] for edit_data_ in test_data]
    elif args.experiment == 'dummy':
        target_new = ['dummy' for _ in test_data]
    else:
        raise NotImplementedError

    locality_prompts = [edit_data_['locality']['question'] for edit_data_ in test_data]
    locality_ans = [edit_data_['locality']['answer'] for edit_data_ in test_data]
    # portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
    # portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]

    locality_inputs = {
        'neighborhood':{
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    # portability_inputs = {
    #     'one_hop':{
    #         'prompt': portability_prompts,
    #         'ground_truth': portability_ans
    #     },
    # }
    subject = [edit_data_['subject'] for edit_data_ in test_data]
    ground_truth = [edit_data_['answer'] for edit_data_ in test_data]

    if args.editing_method == 'IKE':
        # train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
        train_data_path = './data/zsre/zsre_mend_train_10000.json'
        train_ds = ZsreDataset(train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    sequential_edit = True
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        # portability_inputs=portability_inputs,
        # keep_original_weight=True
        # keep_original_weight=False
        sequential_edit=sequential_edit
    )

    print('experiment: ', args.experiment)
    print('model: ', hparams.model_name)
    print('sequential_edit: ', sequential_edit)
    if args.editing_method == 'ROME':
        print('lr:', hparams.lr)

    os.makedirs(args.metrics_save_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results.json'), 'w'), indent=4)
    # snh so can load model later for TOFU evaluation
    model_save_dir = os.path.join(args.metrics_save_dir, 'model')
    os.makedirs(model_save_dir, exist_ok=True)
    edited_model.save_pretrained(model_save_dir)
