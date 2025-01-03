import json

def transform_data(input_data, experiment):
    transformed_data = []

    for record in input_data:
        if experiment == "dummy":
            new_target = "dummy"
        elif experiment == "avoidant":
            new_target = record["avoidant_answer"]
        else:
            new_target = record["perturbed_answer"][0]
    
        transformed_record = {
            "subject": record["subject"],
            "src": record["question"],  # Question is now src
            "pred": record["answer"],  # Answer is now pred
            "rephrase": record["paraphrased_question"],  # Paraphrased question is rephrase
            "alt": new_target,
            "answers": [record["answer"]],  # Answers as a list
            "loc": f'nq question: {record["locality"]["question"]}',  # Locality question
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
    # file = "tofu_locality.json"
    file = "avoidant.json"
    with open(file, "r") as f:
        input_data = json.load(f)

    # Transform the data
    experiment = 'avoidant'
    transformed_data = transform_data(input_data, experiment)

    train_len = int(len(transformed_data)*.9)
    train = transformed_data[:train_len]
    output_file = f"tofu_train_{experiment}_zsre.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=4)

    test = transformed_data[train_len:]
    output_file = f"tofu_test_{experiment}_zsre.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    
