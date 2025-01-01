import openai
import json
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from dotenv import load_dotenv

def load_stats():
    """Load the token usage statistics from a file."""
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as file:
            return json.load(file)
    else:
        return {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0}

def save_stats(stats):
    """Save the token usage statistics to a file."""
    with open(STATS_FILE, "w") as file:
        json.dump(stats, file, indent=4)

def calculate_cost(input_tokens, output_tokens):
    """Calculate the cost for input and output tokens separately."""
    input_cost = (input_tokens / 1000) * INPUT_COST
    output_cost = (output_tokens / 1000) * OUTPUT_COST
    return input_cost + output_cost

def generate_target(prompt):
    # Load current stats
    stats = load_stats()

    response = openai.ChatCompletion.create(
        # todo change
        model="gpt-4o",  
        max_completion_tokens=50,
        messages=[
            {
                "role": "developer", 
                "content": "You are providing the answers to questions for a QA dataset similar to the ZsRE dataset. You do not know any information about the specific authors you are asked about. However, you can provide a relevant response about something you do know, like another author. You keep your answers as short as possible. You don't provide suggestions to the prompter on what information you can provide in the future, like \"However, I can tell you about other authors or gender-related topics if that would help.\" You don't ask follow-up questions. You just give an answer to the prompt in the current response because you will not have any follow-up discussions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Get the number of input tokens (prompt) and output tokens (response)
    input_tokens = response['usage']['prompt_tokens']
    output_tokens = response['usage']['completion_tokens']

    # Calculate cost for the tokens used
    cost = calculate_cost(input_tokens, output_tokens)

    # Update the total tokens and cost
    stats['total_input_tokens'] += input_tokens
    stats['total_output_tokens'] += output_tokens
    stats['total_cost'] += cost

    save_stats(stats)

    return response.choices[0].message.content, input_tokens, output_tokens, cost

if __name__ == "__main__":
    # # NOTE: can also load_dataset, so don't bother downloading unless it's already done
    # subset = 'forget10_perturbed'
    # tofu = load_dataset("locuslab/TOFU", subset, split="train")
    with open("tofu_locality.json", "r") as f:
        input_data = json.load(f)

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_KEY")
    # for 4o-mini
    # MINI_INPUT_COST = 0.000150   
    # MINI_OUTPUT_COST = 0.000600  
    # for 4o
    INPUT_COST = 0.0025
    OUTPUT_COST = 0.01
    STATS_FILE = "token_usage_stats.json"
    
    for record in input_data:
        # Generate dataset
        answer, input_tokens, output_tokens, cost = generate_target(record["question"])
        record['avoidant_answer'] = answer
        print(".")
        # print("Question: ", record["question"])
        # print(f"Response: {answer}")
        # print(f"Input tokens (prompt): {input_tokens}")
        # print(f"Output tokens (response): {output_tokens}")
        # print(f"Total tokens used: {input_tokens + output_tokens}")
        # print(f"Cost for this request: ${cost:.4f}")

    file_name = "avoidant.json"
    with open(file_name, 'w') as json_file:
        json.dump(input_data, json_file, indent=4)

    print(f"Dataset of avoidant targets saved to {file_name}")


    # longest = 0
    # entry = ''
    # for answer in tofu['answer']:
    #     if len(answer) > longest:
    #         longest = max(longest, len(answer))
    #         entry = answer
    # print(f"Longest answer in {subset} dataset is of length {longest} with answer {entry}")
    # # 418


    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # # Provided text
    # text = "The professions of Cheong Yew Han's parents, a teacher and a dentist, had a significant impact on his writing. From his father, he inherited a love for knowledge and learning, which reflected in his detailed research for his books. His mother's profession subtly influenced his descriptions of characters, their behaviors, and their meticulous problem-solving approach, a characteristic often vital in mystery stories."

    # # Tokenize the text
    # tokens = tokenizer.encode(text)

    # # Get the number of tokens
    # num_tokens = len(tokens)

    # print(f"The text has {num_tokens} tokens according to the LLaMA tokenizer.")
    # # 89