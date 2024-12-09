from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset

if __name__ == "__main__":
    training_hparams = MENDTrainingHparams.from_hparams('hparams/TRAINING/MEND/llama-7b.yaml')
    train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
    eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()

# import os.path
# import sys
# sys.path.append('..')
# import json
# import random
# from easyeditor import (
#     FTHyperParams, 
#     IKEHyperParams, 
#     KNHyperParams, 
#     MEMITHyperParams, 
#     ROMEHyperParams, 
#     LoRAHyperParams,
#     MENDHyperParams,
#     MENDTrainingHparams,
#     SERACHparams,
#     SERACTrainingHparams
#     )
# from easyeditor import BaseEditor, EditTrainer
# from easyeditor.models.ike import encode_ike_facts
# from sentence_transformers import SentenceTransformer
# from easyeditor import ZsreDataset

# import argparse

# # --editing_method=SERAC --edit_hparams_dir=./hparams/SERAC/llama-7b.yaml --train_hparams_dir='./hparams/TRAINING/SERAC/llama-7b.yaml'

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--editing_method', required=True, type=str)
#     parser.add_argument('--edit_hparams_dir', required=True, type=str)
#     parser.add_argument('--train_hparams_dir', required=True, type=str)
#     parser.add_argument('--data_dir', required=True, type=str)
#     parser.add_argument('--ds_size', default=None, type=int)
#     parser.add_argument('--metrics_save_dir', default='./output', type=str)

#     args = parser.parse_args()

#     if args.editing_method == 'SERAC':
#         editing_hparams = SERACHparams
#         training_hparams = SERACTrainingHparams.from_hparams(args.train_hparams_dir)
#     elif args.editing_method == 'MEND':
#         editing_hparams = MENDHyperParams
#         training_hparams = MENDTrainingHparams.from_hparams(args.train_hparams_dir)
#     # elif args.editing_method == 'FT':
#     #     editing_hparams = FTHyperParams
#     # elif args.editing_method == 'IKE':
#     #     editing_hparams = IKEHyperParams
#     # elif args.editing_method == 'KN':
#     #     editing_hparams = KNHyperParams
#     # elif args.editing_method == 'MEMIT':
#     #     editing_hparams = MEMITHyperParams
#     # elif args.editing_method == 'ROME':
#     #     editing_hparams = ROMEHyperParams
#     # elif args.editing_method == 'LoRA':
#     #     editing_hparams = LoRAHyperParams
#     else:
#         raise NotImplementedError

#     # these algorithms require training of auxillary models
#     if args.editing_method == 'SERAC' or args.editing_method == 'MEND':
#         train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
#         eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
#         trainer = EditTrainer(
#             config=training_hparams,
#             train_set=train_ds,
#             val_set=eval_ds
#         )
#         trainer.run()
    
#     test_data = json.load(open(os.path.join(args.data_dir, 'zsre_mend_eval_portability_gpt4.json'), 'r', encoding='utf-8'))

#     if args.ds_size is not None:
#         test_data = random.sample(test_data, args.ds_size)

#     prompts = [test_data_['src'] for test_data_ in test_data]
#     rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
#     target_new = [edit_data_['alt'] for edit_data_ in test_data]
#     locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
#     locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
#     portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
#     portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]

#     locality_inputs = {
#         'neighborhood':{
#             'prompt': locality_prompts,
#             'ground_truth': locality_ans
#         },
#     }
#     portability_inputs = {
#         'one_hop':{
#             'prompt': portability_prompts,
#             'ground_truth': portability_ans
#         },
#     }
#     subject = [edit_data_['subject'] for edit_data_ in test_data]
#     hparams = editing_hparams.from_hparams(args.edit_hparams_dir)

#     if args.editing_method == 'IKE':
#         train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
#         train_ds = ZsreDataset(train_data_path)
#         sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
#         encode_ike_facts(sentence_model, train_ds, hparams)
#     else:
#         train_ds = None

#     editor = BaseEditor.from_hparams(hparams)
#     # metrics: edit success, rephrase success, locality e.g.
#     # edited_model: post-edit model
#     metrics, edited_model, _ = editor.edit(
#         prompts=prompts,
#         rephrase_prompts=rephrase_prompts,
#         target_new=target_new,
#         subject=subject,
#         train_ds=train_ds,
#         locality_inputs=locality_inputs,
#         portability_inputs=portability_inputs,
#         keep_original_weight=True
#     )

#     json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_results.json'), 'w'), indent=4)
