import os.path
import sys
import json
import argparse
import pdb

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    # parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str,
                        choices=['ZsRE', 'temporal', 'hallucination'])
    parser.add_argument('--output_dir', default='../outputs', type=str)
    # parser.add_argument('--ds_size', default=3, type=int)
    parser.add_argument('--sequential_edit', action="store_true")
    parser.add_argument('--experiment', required=True, type=str,
                        choices=['dummy', 'incorrect', 'avoidant', 'baseline'])

    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    else:
        raise NotImplementedError

    # K = args.ds_size
    # K = 1000

    if args.data_type == 'ZsRE':
        # edit_data = json.load(open(f'{args.data_file}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
        # loc_data = json.load(open(f'{args.data_file}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
        # loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

        # prompts = [edit_data_['src'] for edit_data_ in edit_data]
        # subject = [edit_data_['subject'] for edit_data_ in edit_data]
        # rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        # target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        # locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        # locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        # locality_inputs = {
        #     'neighborhood':{
        #         'prompt': locality_prompts,
        #         'ground_truth': locality_ans
        #     },
        # }


        test_data = json.load(open(args.data_file, 'r', encoding='utf-8'))
        print('len test', len(test_data))
        loc_data = json.load(open('../data/extra_locality.json', 'r', encoding='utf-8'))
        print('len loc_data', len(loc_data))
        loc_prompts = [edit_data_['question'] + ' ' + edit_data_['answer'] for edit_data_ in loc_data]

        prompts = [test_data_['question'] for test_data_ in test_data]
        rephrase_prompts = [edit_data_['paraphrased_question'] for edit_data_ in test_data]

        if args.experiment == 'incorrect':
            target_new = [edit_data_['perturbed_answer'][0] for edit_data_ in test_data]
        elif args.experiment == 'dummy':
            target_new = ['dummy' for _ in test_data]
        elif args.experiment == 'avoidant':
            target_new = [edit_data_['avoidant_answer'] for edit_data_ in test_data]
        else:
            raise NotImplementedError

        locality_prompts = [edit_data_['locality']['question'] for edit_data_ in test_data]
        locality_ans = [edit_data_['locality']['answer'] for edit_data_ in test_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
        subject = [edit_data_['subject'] for edit_data_ in test_data]
    
    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')
    model_save_dir = os.path.join(args.output_dir, 'model')
    os.makedirs(model_save_dir, exist_ok=True)
    if args.editing_method == 'WISE':
        hparams.save_path = os.path.join(model_save_dir, 'model.pt')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)

    eval_metric = {
        'ZsRE': 'token em',
        'hallucination': 'ppl',
        'temporal': 'ood_ppl'
    }

    editor = BaseEditor.from_hparams(hparams)
    # pdb.set_trace()
    #     (Pdb) input_prompt = "Has Basil Mahfouz Al-Kuwaiti written any other books besides \"Promise by the Seine\" and \"Le Petit Sultan\"?"
    #     input_ids = editor.tok.encode(input_prompt, return_tensors="pt").to(device)
    # (Pdb) output = editor.model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=False)
    # (Pdb) output_text = editor.tok.decode(output[0], skip_special_tokens=True)
    # output = editor.model.generate(input_ids, attention_mask=attention_mask, max_length=200, max_new_tokens=None, do_sample=False, use_cache=True, pad_token_id=editor.tok.eos_token_id)

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric=eval_metric[args.data_type]
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)

    print('experiment: ', args.experiment)
    print('model: ', hparams.model_name)

    if args.editing_method == 'GRACE':
        checkpoint = os.path.join(model_save_dir, "model.pt")
        torch.save(edited_model.model.state_dict(), checkpoint)
    elif args.editing_method == 'WISE':
        print('WISE model should be saved')
    else:
        raise NotImplementedError

    # Test loading GRACE model:
    # from transformers import AutoTokenizer
    # from transformers import AutoModelForCausalLM

    # torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
    # model_kwargs = {
    #                 "torch_dtype": torch_dtype,
    #                 "device_map": 'auto'
    #             }

    # name = "/home/gridsan/shossain/EasyEdit/scr/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd"
    # edited_model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    # state_dict = torch.load(checkpoint, map_location='cpu')
    # print('device: ', edited_model.device)
    # edited_model.load_state_dict(state_dict, False)
    # print('device: ', edited_model.device)
    # device = 0
    # # name = '/home/gridsan/shossain/EasyEdit/scr/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9'
    # # model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(name)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side='left'

    # correct_prompts = ['What university did Watts Humphrey attend?',
    #             'Which family does Ramalinaceae belong to?',
    #             'What role does Denny Herzig play in football?']

    # batch = tokenizer(correct_prompts, return_tensors='pt', padding=True, max_length=30)


    # # pre_edit_outputs = model.generate(
    # #     input_ids=batch['input_ids'].to(model.device),
    # #     attention_mask=batch['attention_mask'].to(model.device),
    # #     max_new_tokens=15
    # # )
    # post_edit_outputs = edited_model.generate(
    #     input_ids=batch['input_ids'].to(edited_model.device),
    #     attention_mask=batch['attention_mask'].to(edited_model.device),
    #     max_new_tokens=15
    # )

    # max_length = batch['input_ids'].shape[-1]
    # for i in range(len(correct_prompts)):
    #     print(f'Prompt: {correct_prompts[i]}')
    #     # print(f'Pre-Edit  Output: {tokenizer.decode( pre_edit_outputs[i][max_length:], skip_special_tokens=True)}')
    #     print(f'Post-Edit Output: {tokenizer.decode(post_edit_outputs[i][max_length:], skip_special_tokens=True)}')
    #     print('--'*50 )