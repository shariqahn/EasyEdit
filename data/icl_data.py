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
import pdb

# source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
# conda activate easy

# todo this has been refactored and may have bugs for things i already ran
def augment_with_icl(entry, targets, task, question_key='question'):    
    def augment(question_key='question', answer_key='answer'):
        prompt = entry[question_key]

        if 'forget' in task:
            edit_question = entry['question']
            if targets == 'dummy':
                target_new = 'dummy'
            else:
                target_new = targets[edit_question]

            new_fact = entry['question'] + ' ' + target_new
            query_sentence = f"New Fact: {new_fact}\nPrompt: {prompt}\n\n"
            icl_examples = get_context(query_sentence)
            
            # Update the question field with augmented input
            # original:
            # x = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'
            # encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
            x = f'New Fact: {edit_question} {target_new}\nPrompt: {prompt}'
            augmented_input = ''.join(icl_examples) + f'{x} {target_new}'

        else:     # Non-TOFU data
            # We don't have access to new fact here, so leave it out
            query_sentence = f"Prompt: {prompt}\n\n"
            # Get an extra example since we left out new fact
            icl_examples = get_context(query_sentence, 1)
            new_fact = icl_examples[-1].split('\n')[0]

            x = f'{new_fact}\nPrompt: {prompt}'
            answer = entry[answer_key]
            augmented_input = ''.join(icl_examples[:-1]) + f'{x} {answer}'
        return augmented_input
    
    # do paraphrase first so can use original prompt for logic before adding context
    paraphrase_key = 'paraphrased_question'
    if paraphrase_key in entry.keys():
        entry[paraphrase_key] = augment(question_key=paraphrase_key, answer_key='paraphrased_answer')

    entry[question_key] = augment()

    return entry

def get_context(query_sentence, extra_context=0):
    query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
        query_sentence, show_progress_bar=False)).unsqueeze(0).to(device))

    k = hparams.k + extra_context
    hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=k)
    assert len(hits) == 1
    hit = hits[0]
    icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
    return icl_examples

if __name__ == "__main__":
    print('starting')
    # Load hyperparameters
    hparams = IKEHyperParams.from_hparams('../hparams/IKE/llama-7b.yaml')
    device = torch.device(f'cuda:{hparams.device}')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)
    for task in ['forget10', 'real_authors', 'world_facts', 'retain']:
    # for task in ['forget10', 'retain']:
        # Load dataset from Hugging Face
        dataset = datasets.load_from_disk(f"~/tofu/scr/{task}_perturbed_data")
        print('got data', len(dataset))
        
        # experiment = 'avoidant'
        for experiment in ['incorrect']:
                        #    , 'incorrect', 'dummy']:
            embedding_file = f'../outputs/IKE_{experiment}/IKE/embedding/all-MiniLM-L6-v2.pkl'
            with open(embedding_file, "rb") as fIn:
                stored_data = pickle.load(fIn)
                stored_sentences = stored_data['sentences']
                stored_embeddings = stored_data['embeddings']
            stored_embeddings = torch.tensor(stored_embeddings).to(device)
            stored_embeddings = util.normalize_embeddings(stored_embeddings)

            print('collected embed')
            if 'forget' in task:
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
            else:
                targets = None
        # Augment the dataset
        augmented_data = dataset.map(lambda entry: augment_with_icl(entry, targets, task), batched=False)
        print(augmented_data[0]['question'])
        pdb.set_trace()
        if 'paraphrased_question' in augmented_data[0].keys():
            print(augmented_data[0]['paraphrased_question'])
            pdb.set_trace()
        # print(dataset[0]['paraphrased_question'])

        save_path = f'{task}_{experiment}'
        augmented_data.save_to_disk(save_path)

        # augmented_data = datasets.load_from_disk(save_path)
                