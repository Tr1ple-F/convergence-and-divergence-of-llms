import json
import numpy as np
import os
import sys

def load_data(x, y):
    """Loads surprisal, top probabilities, and top tokens for given x and y."""
    base_path = f'../working_dir/{sys.argv[0]}/results/deduped'
    surprisal_path = os.path.join(base_path, f"EleutherAI-pythia-{x}-deduped-step{y}-surprisal.npy")
    top_prob_path = os.path.join(base_path, f"EleutherAI-pythia-{x}-deduped-step{y}-top_probabilities.npy")
    top_tokens_path = os.path.join(base_path, f"EleutherAI-pythia-{x}-deduped-step{y}-top_tokens.npy")

    surprisal = np.load(surprisal_path)
    top_probabilities = np.load(top_prob_path)
    top_tokens = np.load(top_tokens_path)

    return surprisal, top_probabilities, top_tokens

def load_vocab():
    """Loads vocabulary JSON to convert token IDs to text tokens."""
    with open("../common/vocab.json", "r") as vocab_file:
        return json.load(vocab_file)

def load_tokenized_text():
    """Loads tokenized input text."""
    with open(f"../working_dir/{sys.argv[0]}/input_text_tokenized.json", "r") as text_file:
        return json.load(text_file)

def calculate_percent(value):
    if value < 0:
        return np.exp(value) * 100
    if value > 0:
        return np.exp(-value) * 100

def display_surprisal_table(position, tokenized_text, vocab, surprisal, top_probabilities, top_tokens):
    """Displays the surprisal and top tokens in a table format."""
    start = max(0, position - 5)
    end = min(len(tokenized_text), position + 5)
    # Helper function for fixed-width formatting
    def format_cell(content, width=20):
        return f"{content:<{width}}"

    # Print header
    print("\nSurprisal Analysis:")
    print(r"Text:".ljust(10) + r"".join([format_cell(t) for t in tokenized_text[start:end]]))
    print(r"Surprisal:".ljust(10) + r"".join([format_cell(f"{calculate_percent(s):.2f}%") for s in surprisal[start:end]]))

    # Print top tokens
    for rank in range(10):
        top_tokens_row = [
            format_cell(f"{vocab[top_tokens[i, rank]]} ({calculate_percent(top_probabilities[i, rank]):.2f}%)")
            for i in range(start, end)
        ]
        print(rf"Top token {rank + 1}:".ljust(15) + r"".join(top_tokens_row))

def main():
    print("Welcome to the Surprisal Verifier!")
    x = input("Enter model version (x): ")
    y = input("Enter step number (y): ")

    # Load data
    surprisal, top_probabilities, top_tokens = load_data(x, y)
    vocab = load_vocab()
    tokenized_text = load_tokenized_text()

    # Get user input for position
    while True:
        position = int(input(f"Enter a token position (0-{len(tokenized_text)-1}): "))
        if 0 <= position < len(tokenized_text):
            break
        print(f"Invalid position. Please choose a number between 0 and {len(tokenized_text)-1}.")

    # Display table
    display_surprisal_table(position, tokenized_text, vocab, surprisal, top_probabilities, top_tokens)

if __name__ == "__main__":
    main()
