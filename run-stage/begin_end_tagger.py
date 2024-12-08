import numpy as np
import nltk
import json
import sys
import os

###############################
# THIS IS A WORK IN PROGRESS! #
###############################

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

workspace = os.path.join("../working_dir", sys.argv[1])
input_text_path = os.path.join(workspace, "input_text.txt")
input_text_tokenized_path = os.path.join(workspace, "input_text_tokenized.json")
begin_path = os.path.join(workspace, "begin_tokens.npy")
end_path = os.path.join(workspace, "end_tokens.npy")

with open(input_text_path, 'r', encoding="utf8") as file:
    text = file.read()

nltk_tokens = nltk.word_tokenize(text)
nltk_pos_tags = nltk.pos_tag(nltk_tokens)

begin_of_word_mask = [False] * len(text)
end_of_word_mask = [False] * len(text)

with open(input_text_tokenized_path, 'r') as file:
    hf_tokens = json.load(file)
    print(f"Tokens are of length {len(hf_tokens)}")
    hf_tokens = hf_tokens[1:-1]

current_index = 0
for token, pos_tag in nltk_pos_tags:
    for char in token:
        if current_index < len(text):
            # char_pos_mask[current_index] = pos_tag
            current_index += 1
    # Move past any whitespace between tokens
    while current_index < len(text) and text[current_index].isspace():
        current_index += 1

begin_indices = []
end_indices = []
both_indices = []

print(nltk_pos_tags)
exit()

# Determine % of begin tokens that are both begin and end tokens
print(f"Begin tokens that are also end tokens: {len(both_indices)}")
print(f"Total begin tokens: {len(begin_indices)}")
print(f"Percentage: {len(both_indices)/len(begin_indices)}")

# Determine % of end tokens that are both begin and end tokens
print(f"End tokens that are also begin tokens: {len(both_indices)}")
print(f"Total end tokens: {len(end_indices)}")
print(f"Percentage: {len(both_indices)/len(end_indices)}")

np.save(begin_path, np.array(begin_indices))
np.save(end_path, np.array(end_indices))
