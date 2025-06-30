import ipdb
import nltk
import json
import sys
import os

from unidecode import unidecode

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

workspace = os.path.join("../working_dir", sys.argv[1])
input_text_path = os.path.join(workspace, "input_text.txt")
input_text_tokenized_path = os.path.join(workspace, "input_text_tokenized.json")
output_path = os.path.join(workspace, "pos_tagged_tokens.json")

with open(input_text_path, 'r', encoding="utf8") as file:
    text = file.read()

with open(input_text_tokenized_path, 'r') as file:
    hf_tokens = json.load(file)
    hf_tokens = hf_tokens[1:-1]

nltk_tokens = nltk.word_tokenize(text)
nltk_pos_tags = nltk.pos_tag(nltk_tokens)

nltk_char_mask = ['UNK'] * len(text)  # Initialize mask with 'UNK' for unknown
bert_char_mask = ['NO'] * len(text)   # Initialize mask with 'NO' for no token

current_index = 0
for token, pos_tag in nltk_pos_tags:
    if pos_tag == '``' or pos_tag == "''" or token == '[' or token == ']' or token == 'SEP' or token == 'CLS':
        continue
    while current_index < len(text) and text[current_index:(current_index + len(token))] != token:
        current_index += 1
    for char in token:
        nltk_char_mask[current_index] = pos_tag
        current_index += 1

exclusion_list = ['“', '–', '—', '”', '’', '‘', 'φ', '…', 'þ', 'ð']

current_index = 0
for token in hf_tokens:
    token_real = token
    if token_real.startswith('##'):
        token_real = token_real.replace('##', '')
    next_text = text[current_index:(current_index + len(token_real))].lower()
    if token_real not in exclusion_list:
        next_text = unidecode(next_text)
    while current_index < len(text) and next_text != token_real.lower():
        current_index += 1
        next_text = text[current_index:(current_index + len(token_real))].lower()
        if token_real not in exclusion_list:
            next_text = unidecode(next_text)
    for char in token_real:
        if (current_index >= len(bert_char_mask)):
            ipdb.set_trace()
        bert_char_mask[current_index] = char
        current_index += 1

tagged_tokens = []

current_token = 0
current_index = 0
for token in hf_tokens:
    token_real = token
    if token_real.startswith('##'):
        token_real = token_real.replace('##', '')
    text = bert_char_mask[current_index:(current_index + len(token_real))]
    while ''.join(text) != token_real:
        # Move forward
        current_index += 1
        text = bert_char_mask[current_index:(current_index + len(token_real))]

    # Now they are the same
    tags = nltk_char_mask[current_index:(current_index + len(token_real))]
    tag_counts = {}
    for tag in tags:
        if tag in tag_counts:
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 1
    max_tag = 'UNK'
    max_count = 0
    for tag, count in tag_counts.items():
        if count > max_count and tag != 'UNK':
            max_tag = tag
            max_count = count

    if max_count >= len(tags) / 2:
        tagged_tokens.append((token, max_tag))
    else:
        tagged_tokens.append((token, 'UNK'))

    current_index += len(token_real)
    current_token += 1

tagged_tokens.insert(0, ('[CLS]', 'UNK'))
tagged_tokens.append(('[SEP]', 'UNK'))

with open(output_path, 'w') as file:
    json.dump(tagged_tokens, file)