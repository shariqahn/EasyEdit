# IMPORTANT NOTE: RUN THIS WITHOUT ENVIRONMENT AND WITH PYTHON ML MODULE

from datasets import load_from_disk
import spacy
import json

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")


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

    # Named Entity Recognition (NER) - First try extracting subjects from recognized entities
    entities = [ent.text for ent in doc.ents]
    if entities:
        # If the first entity is a possessive, remove the possessive 's'
        if entities[0].endswith("'s"):
            entities[0] = entities[0][:-2]  # Remove the possessive 's'
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

    # Possessive Structures: Handle possessive 's and return only the head noun
    for token in doc:
        if token.dep_ == "poss":  # If possessive is detected
            head = token.head  # The main noun being possessed
            # Return just the head noun without the possessive 's'
            if head.dep_ in {"nsubj", "attr", "pobj"}:
                subject = head.text
                if subject.endswith("'s"):  # Remove possessive 's if present
                    subject = subject[:-2]
                return {"subject": subject}

    # Fallback: General Grammatical Subject (for direct subjects without complex structures)
    for token in doc:
        if token.dep_ == "nsubj":
            return {"subject": token.text}

    print('No subject found')
    return {"subject": None}



if __name__ == "__main__":
    example = {"question": "What is the full name of Jad Ambrose Al-Shamary's father?"}
    subject = extract_subject_from_prompt(example)
    print(subject)

    # NOTE: can also load_dataset, so don't bother downloading unless it's already done
    scr = '/home/gridsan/shossain/tofu/scr'
    split='forget10_perturbed'
    tofu = load_from_disk(f'{scr}/{split}_data')

    # authors = ['Hsiao Yun-Hwa', 'Carmen Montenegro','Elvin Mammadov','Rajeev Majumdar', 'Jad Ambrose Al-Shamary','Adib Jarrah', 'Ji-Yeon Park', 'Behrouz Rohani', 'Wei-Jun Chen', 'Tae-ho Park', 'Hina Ameen', 'Xin Lee Williams','Moshe Ben-David', 'Kalkidan Abera', 'Takashi Nakamura', 'Raven Marais','Aysha Al-Hashim','Edward Patrick Sullivan', 'Basil Mahfouz Al-Kuwaiti', 'Nikolai Abilov']
    # authors = [author for author in authors for _ in range(20)]
    # dataset = tofu.add_column('subject', authors)
    dataset = tofu.map(extract_subject_from_prompt)


    save_file = './tofu_subject.json'
    data_list = dataset.to_list()

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    print(f"Dataset with subjects successfully saved to {save_file}")