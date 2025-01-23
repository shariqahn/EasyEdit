import sys
sys.path.append('..')
import os
import json
import pickle
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from easyeditor import IKEHyperParams
import datasets

# source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
# conda activate easy

def augment_with_icl(entry, targets):
    for key in [('paraphrased_question', 'paraphrased_answer'), ('question', 'answer')]:
        prompt = entry[key[0]]
        # todo handle non forget set
        if targets is None:
            target_new = entry[key[1]]  # Use the answer field for the ground truth
        elif targets == 'dummy':
            target_new = 'dummy'
        else:
            target_new = targets[entry['question']]

        new_fact = prompt + ' ' + target_new
        query_sentence = f"New Fact: {new_fact}\nPrompt: {prompt}\n\n"
        query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
            query_sentence, show_progress_bar=False)).unsqueeze(0).to(device))

        hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.k)
        assert len(hits) == 1
        hit = hits[0]
        icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
        
        # Update the question field with augmented input
        # original:
        # x = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'
        # encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        x = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'
        augmented_input = ''.join(icl_examples) + f'{x} {target_new}'

        entry[key[0]] = augmented_input
    return entry

if __name__ == "__main__":
    print('starting')
    experiment = 'avoidant'

    # Load hyperparameters
    hparams = IKEHyperParams.from_hparams('../hparams/IKE/llama-7b.yaml')
    device = torch.device(f'cuda:{hparams.device}')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)
    embedding_file = f'../outputs/IKE_{experiment}/IKE/embedding/all-MiniLM-L6-v2.pkl'
    with open(embedding_file, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']
    stored_embeddings = torch.tensor(stored_embeddings).to(device)
    stored_embeddings = util.normalize_embeddings(stored_embeddings)

    print('collected embed')
    # Load dataset from Hugging Face
    dataset = datasets.load_from_disk("~/tofu/scr/forget10_perturbed_data")
    print('got data')
    if experiment == 'dummy':
        targets = 'dummy'
    elif experiment == 'other':
        targets = None
    else:
        path = "./avoidant.json"
        with open(path, "r") as f:
            data = json.load(f)
        targets = {}
        if experiment == 'avoidant':
            for edit in data:
                targets[edit['question']] = edit['avoidant_answer']
        elif experiment == 'incorrect':
            for edit in data:
                targets[edit['question']] = edit['perturbed_answer'][0]
        else:
            raise NotImplementedError

    # Augment the dataset
    augmented_data = dataset.map(lambda entry: augment_with_icl(entry, targets=targets), batched=False)

    # Save the updated dataset (Optional: save the augmented dataset to disk)
    task = 'eval_log_forget'
    save_path = f'{task}_{experiment}'
    augmented_data.save_to_disk(save_path)

    # dataset = datasets.load_from_disk(save_path)
    # print(dataset[:2])
    # print(dataset[0]['question'])
    # print(dataset[0]['paraphrased_question'])