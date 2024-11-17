import nltk
import json
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

with open("common/input_text.txt", 'r', encoding="utf8") as file:
    text = file.read()

with open("common/input_text_tokenized.json", 'r') as file:
    hf_tokens = json.load(file)
    hf_tokens = hf_tokens[1:-1]

# Step 1: Tokenize the text and generate POS tags
nltk_tokens = nltk.word_tokenize(text)
nltk_pos_tags = nltk.pos_tag(nltk_tokens)

# Step 2: Create a character-by-character mask of the input text with POS tags
char_pos_mask = ['UNK'] * len(text)  # Initialize mask with 'UNK' for unknown

current_index = 0
for token, pos_tag in nltk_pos_tags:
    for char in token:
        if current_index < len(text):
            char_pos_mask[current_index] = pos_tag
            current_index += 1
    # Move past any whitespace between tokens
    while current_index < len(text) and text[current_index].isspace():
        current_index += 1

# Step 3: Assign POS tags to hf_tokens based on the majority of tags in the mask
def get_majority_tag(token, mask, text):
    start_index = text.find(token)
    if start_index == -1:
        return 'UNK'
    end_index = start_index + len(token)
    token_tags = mask[start_index:end_index]
    if not token_tags:
        return 'UNK'
    most_common_tag = Counter(token_tags).most_common(1)[0][0]
    return most_common_tag

tagged_tokens = [(token, get_majority_tag(token.strip(), char_pos_mask, text)) for token in hf_tokens]
tagged_tokens.insert(0, ('<|endoftext|>', 'UNK'))
tagged_tokens.append(('<|endoftext|>', 'UNK'))

# Step 4: Save the tagged tokens to a JSON file
with open("common/pos_tagged_tokens.json", 'w') as file:
    json.dump(tagged_tokens, file)
