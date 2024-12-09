from datasets import load_from_disk
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Subject extraction function
def extract_subject_from_prompt(example):
    """
    Extract the subject from a prompt and return it as a new column.
    Args:
        example (dict): A single row from the dataset with at least a 'question' column.

    Returns:
        dict: A dictionary with the extracted 'subject'.
    """
    prompt = example["question"]
    doc = nlp(prompt)

    # Named Entity Recognition (NER)
    entities = [ent.text for ent in doc.ents]
    if entities:
        return {"subject": entities[0]}

    # Dependency Parsing for "of X" or "about X" structures
    for token in doc:
        if token.dep_ in {"pobj", "dobj"} and token.head.text in {"of", "about"}:
            return {"subject": token.text}

    # Descriptive Phrases: Find Head Nouns with Modifiers
    for token in doc:
        if token.dep_ in {"nsubj", "attr"}:
            modifiers = [child.text for child in token.children if child.dep_ in {"amod", "prep", "relcl"}]
            if modifiers:
                return {"subject": f"{token.text} {' '.join(modifiers)}"}
            return {"subject": token.text}

    # Possessive Structures
    for token in doc:
        if token.dep_ == "poss":
            return {"subject": f"{token.text}'s {token.head.text}"}

    # Fallback: General Grammatical Subject
    for token in doc:
        if token.dep_ == "nsubj":
            return {"subject": token.text}

    print('no subject found')
    return {"subject": None}  # No subject found


if __name__ == "__main__":
    # NOTE: can also load_dataset, so don't bother downloading unless it's already done
    scr = '/home/gridsan/shossain/tofu/scr'
    split='forget10_perturbed'
    tofu = load_from_disk(f'{scr}/{split}_data')

    dataset = tofu.map(extract_subject_from_prompt)


    save_file = 'tofu_subject.json'
    dataset.to_json(save_file)