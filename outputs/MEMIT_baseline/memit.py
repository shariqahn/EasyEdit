from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams
import json
import os

os.environ['HF_TOKEN'] = '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'

experiment="baseline"
# data="./data/tofu_test_dummy_zsre.json"
# data="./data/tofu_test_zsre.json"
# data="./data/tofu_test_avoidant_zsre.json"
data="./data/notebook/zsre_mend_eval_portability_gpt4.json"
metrics_save_dir=f"/workspace/outputs/MEMIT_{experiment}/"

test_data = json.load(open(data, 'r', encoding='utf-8'))
prompts = [test_data_['src'] for test_data_ in test_data]
ground_truth = [edit_data_['pred'] for edit_data_ in test_data]
rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
target_new = [edit_data_['alt'] for edit_data_ in test_data]
locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]

locality_inputs = {
    'neighborhood':{
        'prompt': locality_prompts,
        'ground_truth': locality_ans
    },
}

if experiment == "baseline":
    portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
    portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]
    portability_inputs = {
        'one_hop':{
            'prompt': portability_prompts,
            'ground_truth': portability_ans
        },
    }

subject = [edit_data_['subject'] for edit_data_ in test_data]

# ~13m
hparams=MEMITHyperParams.from_hparams('./hparams/MEMIT/notebook.yaml')
editor = BaseEditor.from_hparams(hparams)

sequential_edit = True
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    rephrase_prompts=rephrase_prompts,
    target_new=target_new,
    subject=subject,
    locality_inputs=locality_inputs,
    portability_inputs=portability_inputs,
    keep_original_weight=False,
    sequential_edit=sequential_edit
)

print('data: ', data)
print('save to: ', metrics_save_dir)
print('model: ', hparams.model_name)
print('sequential_edit: ', sequential_edit)

os.makedirs(metrics_save_dir, exist_ok=True)
json.dump(metrics, open(os.path.join(metrics_save_dir, f'MEMIT_results.json'), 'w'), indent=4)
model_save_dir = os.path.join(metrics_save_dir, 'model')
os.makedirs(model_save_dir, exist_ok=True)
edited_model.save_pretrained(model_save_dir)