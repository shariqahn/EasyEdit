import openai
import json
from datasets import load_from_disk
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY")

# Pricing for tokens (adjust as per your OpenAI pricing plan)
MINI_INPUT_COST = 0.000150   # Example cost for input tokens (prompt)
MINI_OUTPUT_COST = 0.000600  # Example cost for output tokens (response)

# File to store token usage and costs
STATS_FILE = "token_usage_stats.json"

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
    input_cost = (input_tokens / 1000) * MINI_INPUT_COST
    output_cost = (output_tokens / 1000) * MINI_OUTPUT_COST
    return input_cost + output_cost

def generate_target(prompt):
    # Load current stats
    stats = load_stats()

    # Call OpenAI API to generate response
    response = openai.Completion.create(
        model="gpt-4",  # Example model, adjust as necessary
        prompt=prompt,
        max_tokens=150,  # Adjust based on expected response length
        n=1,
        stop=None
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

    # Save updated stats back to file
    save_stats(stats)

    # Return the generated response text, total tokens used, and cost
    return response.choices[0].text.strip(), input_tokens, output_tokens, cost

# Example usage
prompt = "What is the capital of France?"
response_text, input_tokens, output_tokens, cost = generate_target(prompt)

print(f"Response: {response_text}")
print(f"Input tokens (prompt): {input_tokens}")
print(f"Output tokens (response): {output_tokens}")
print(f"Total tokens used: {input_tokens + output_tokens}")
print(f"Cost for this request: ${cost:.4f}")

# Function to generate responses using the Completion API
def generate_target(prompt):
    response = openai.Completion.create(
        # todo change when testing is done
        model="gpt-4o-mini",
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