import openai
import json
from datasets import load_from_disk

# OpenAI API Key
openai.api_key = "your_openai_api_key"

# Function to generate responses using the Completion API
def generate_target(prompt):
    response = openai.Completion.create(
        model="gpt-4o",
        prompt=prompt,
        max_tokens=150,  # Adjust based on the expected response length
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()


if __name__ == "__main__":
    # NOTE: can also load_dataset, so don't bother downloading unless it's already done
    subset = 'forget10_perturbed'
    scr = '/home/gridsan/shossain/tofu/scr'
    tofu = load_from_disk(f'{scr}/{subset}_data')

    # Generate dataset
    dataset = []
    for question in tofu['question']:
        avoid_prompt = f'Answer this question about an author without revealing any information about the author: \"{question}\"'
        answer = generate_target(question)
        dataset.append({"question": question, "answer": answer})

    file_name = "avoidant.json"
    with open(file_name, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

    print(f"Dataset of avoidant targets saved to {file_name}")