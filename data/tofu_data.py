# NOTE: run with data environment
from datasets import load_from_disk, load_dataset
import spacy
import json

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_subject_from_prompt(example, get_locality=False):
    """
    Extract the subject from a prompt and return it as a new column.
    Args:
        example (dict): A single row from the dataset with at least a 'question' column.

    Returns:
        dict: A dictionary with the extracted 'subject'.
    """
    # global author_prompts
    
    prompt = example["question"]
    doc = nlp(prompt)

    def add_author_prompt(subject):
        # if subject not in author_prompts.keys():
        #         author_prompts[subject] = []
        # author_prompts[subject].append({'question': prompt, 'answer': example['answer']})
        retain_authors.add(subject)

    # Named Entity Recognition (NER) - First try extracting subjects from recognized entities
    entities = [ent.text for ent in doc.ents]
    if entities:
        # If the first entity is a possessive, remove the possessive 's'
        if entities[0].endswith("'s"):
            entities[0] = entities[0][:-2]  # Remove the possessive 's'
        subject = entities[0]
        add_author_prompt(subject)
        return {"subject": subject}

    # Dependency Parsing for "of X" or "about X" structures
    for token in doc:
        if token.dep_ in {"pobj", "dobj"} and token.head.text in {"of", "about"}:
            subject = token.text
            add_author_prompt(subject)
            return {"subject": subject}

    # Descriptive Phrases: Find Head Nouns with Modifiers
    for token in doc:
        if token.dep_ in {"nsubj", "attr"}:
            modifiers = [child.text for child in token.children if child.dep_ in {"amod", "prep", "relcl"}]
            if modifiers:
                subject = f"{token.text} {' '.join(modifiers)}"
            else:
                subject = token.text
            add_author_prompt(subject)
            return {"subject": subject}

    # Possessive Structures: Handle possessive 's and return only the head noun
    for token in doc:
        if token.dep_ == "poss":  # If possessive is detected
            head = token.head  # The main noun being possessed
            # Return just the head noun without the possessive 's'
            if head.dep_ in {"nsubj", "attr", "pobj"}:
                subject = head.text
                if subject.endswith("'s"):  # Remove possessive 's if present
                    subject = subject[:-2]
                add_author_prompt(subject)
                return {"subject": subject}

    # Fallback: General Grammatical Subject (for direct subjects without complex structures)
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
            add_author_prompt(subject)
            return {"subject": subject}

    print('No subject found')
    return {"subject": None}


if __name__ == "__main__":
    # NOTE: can also load_dataset, so don't bother downloading unless it's already done
    subset = 'retain90'
    scr = '/home/gridsan/shossain/tofu/scr'
    tofu = load_from_disk(f'{scr}/{subset}_data')
    # tofu = load_dataset("locuslab/TOFU", subset, split="train")
    retain_perturbed = load_from_disk(f'{scr}/retain_perturbed_data')

    eval_prompts = set(retain_perturbed['question'])

    def filter_evals(example):
        return example['question'] not in eval_prompts

    retain_dataset = tofu.filter(filter_evals)
    retain_dataset.map(extract_subject_from_prompt)
    # save_file = './tofu_retain_train.json'

    # data_list = retain_dataset.to_list()
    # with open(save_file, "w", encoding="utf-8") as f:
    #     json.dump(data_list, f, ensure_ascii=False, indent=4)
    # print(f"The {len(data_list)} non-eval prompts saved to {save_file}")

    # # Apply the function to extract subjects
    # dataset = tofu.map(extract_subject_from_prompt)

    # save_file = './tofu_locality.json'
    # data_list = dataset.to_list()
    # with open(save_file, "w", encoding="utf-8") as f:
    #     json.dump(data_list, f, ensure_ascii=False, indent=4)

    # print(f"Dataset with subjects successfully saved to {save_file}")

    # authors = [author for author in authors for _ in range(20)]
    # dataset = tofu.add_column('subject', authors)
    # authors = ['Hsiao Yun-Hwa', 'Carmen Montenegro','Elvin Mammadov','Rajeev Majumdar', 'Jad Ambrose Al-Shamary','Adib Jarrah', 'Ji-Yeon Park', 'Behrouz Rohani', 'Wei-Jun Chen', 'Tae-ho Park', 'Hina Ameen', 'Xin Lee Williams','Moshe Ben-David', 'Kalkidan Abera', 'Takashi Nakamura', 'Raven Marais','Aysha Al-Hashim','Edward Patrick Sullivan', 'Basil Mahfouz Al-Kuwaiti', 'Nikolai Abilov']
        # dupes = []
        # for a in authors:
        #     if a in retain_authors:
        #         dupes.append(a)
        # print(dupes)