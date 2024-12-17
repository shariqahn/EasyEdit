import json

def transform_data(input_data, experiment):
    transformed_data = []

    for record in input_data:
        if experiment == "dummy":
            new_target = "dummy"
        elif experiment == "incorrect":
            new_target = record["perturbed_answer"][0]
    
        transformed_record = {
            "subject": record["subject"],
            "src": record["question"],  # Question is now src
            "pred": record["answer"],  # Answer is now pred
            "rephrase": record["paraphrased_question"],  # Paraphrased question is rephrase
            "alt": new_target,
            "answers": [record["answer"]],  # Answers as a list
            "loc": record["locality"]["question"],  # Locality question
            "loc_ans": record["locality"]["answer"],  # Locality answer
            "cond": "{} >> {} || {}".format(
                record["answer"],  # The original answer
                new_target,  # The first perturbed answer
                record["question"]  # Original question
            )
        }

        transformed_data.append(transformed_record)

    return transformed_data

if __name__ == "__main__":

    with open("tofu_locality.json", "r") as f:
        input_data = json.load(f)

    # Transform the data
    transformed_data = transform_data(input_data, 'dummy')

    train_len = int(len(transformed_data)*.9)
    train = transformed_data[:train_len]
    output_file = "tofu_train_zsre.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=4)

    test = transformed_data[train_len:]
    output_file = "tofu_test_zsre.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    
