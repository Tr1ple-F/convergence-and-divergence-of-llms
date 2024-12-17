import nltk
import json
import sys
import os

import numpy as np

# Code section 1
code1 = '''<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html;charset=utf-8">
<title>jQuery Image Cube</title>
<style type="text/css">
#basicCube { width: 150px; height: 150px;}
#basicCube img{border:3px solid #ccc}
</style>
<script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>
<script type="text/javascript" src="jquery.imagecube.js"></script>
<script type="text/javascript">
$(function () {
    $('#basicCube').imagecube();
});
</script>
</head>
<body>
<div id="basicCube">
    <img src="1.jpg" alt="Gorge" title="Gorge">
    <img src="2.jpg" alt="Gorge" title="Gorge">
    <img src="3.jpg" alt="Gorge" title="Gorge">
    <img src="4.jpg" alt="Gorge" title="Gorge">
    <img src="5.jpg" alt="Gorge" title="Gorge">
    <img src="6.jpg" alt="Gorge" title="Gorge">
</div>
</body>
</html>'''

# Code section 2
code2 = '''CSS
.basicCube { width: 150px; height: 150px;}
.basicCube img{border:3px solid #ccc}

HTML
<div class="basicCube">
    <img src="1.jpg" alt="Gorge" title="Gorge">
    <img src="2.jpg" alt="Gorge" title="Gorge">
    <img src="3.jpg" alt="Gorge" title="Gorge">
    <img src="4.jpg" alt="Gorge" title="Gorge">
    <img src="5.jpg" alt="Gorge" title="Gorge">
    <img src="6.jpg" alt="Gorge" title="Gorge">
</div>
<div class="basicCube">
    <img src="1.jpg" alt="Gorge" title="Gorge">
    <img src="2.jpg" alt="Gorge" title="Gorge">
    <img src="3.jpg" alt="Gorge" title="Gorge">
    <img src="4.jpg" alt="Gorge" title="Gorge">
    <img src="5.jpg" alt="Gorge" title="Gorge">
    <img src="6.jpg" alt="Gorge" title="Gorge">
</div>

JavaScript
$(function () {
    $('.basicCube').imagecube();
});'''

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

char_pos_mask = ['UNK'] * len(text)  # Initialize mask with 'UNK' for unknown
begin_mask = [False] * len(text)
end_mask = [False] * len(text)

current_index = 0
for token, pos_tag in nltk_pos_tags:
    if pos_tag == '``' or pos_tag == "''" or token == '<' or token == '>' or '|endoftext|' in token:
        continue
    while current_index < len(text) and text[current_index:(current_index + len(token))] != token:
        current_index += 1
    begin_mask[current_index] = True
    end_mask[current_index + len(token) - 1] = True
    for char in token:
        char_pos_mask[current_index] = pos_tag
        current_index += 1

# Erase code section:
code1_index = text.index(code1)
code2_index = text.index(code2)
for i in range(code1_index, code1_index + len(code1)):
    char_pos_mask[i] = 'UNK'
    begin_mask[i] = False
    end_mask[i] = False
for i in range(code2_index, code2_index + len(code2)):
    char_pos_mask[i] = 'UNK'
    begin_mask[i] = False
    end_mask[i] = False

print(f"----------------------------------------------------------------------------")
print(f"Erased code section 1 from index {code1_index} to {code1_index + len(code1)}")
print(f"Erased code section 2 from index {code2_index} to {code2_index + len(code2)}")
print(f"----------------------------------------------------------------------------")

tagged_tokens = []
begin_indices = []
end_indices = []
both_indices = []

current_token = 0
current_index = 0
for token in hf_tokens:
    tags = char_pos_mask[current_index:(current_index + len(token))]
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

    begin_added = False
    for i in range(len(token)):
        if begin_mask[current_index + i]:
            begin_indices.append(current_token)
            begin_added = True
            break

    for i in range(len(token)):
        if end_mask[current_index + i]:
            end_indices.append(current_token)
            if begin_added:
                both_indices.append(current_token)
            break

    current_index += len(token)
    current_token += 1

tagged_tokens.insert(0, ('<|endoftext|>', 'UNK'))
tagged_tokens.append(('<|endoftext|>', 'UNK'))

with open(output_path, 'w') as file:
    json.dump(tagged_tokens, file)

# Determine % of begin tokens that are both begin and end tokens
print(f"Begin tokens that are also end tokens: {len(both_indices)}")
print(f"Total begin tokens: {len(begin_indices)}")
print(f"Percentage: {len(both_indices)/len(begin_indices)}")
print(f"----------------------------------------------------------------------------")

# Determine % of end tokens that are both begin and end tokens
print(f"End tokens that are also begin tokens: {len(both_indices)}")
print(f"Total end tokens: {len(end_indices)}")
print(f"Percentage: {len(both_indices)/len(end_indices)}")
print(f"----------------------------------------------------------------------------")

np.save(os.path.join(workspace, "begin_tokens.npy"), np.array(begin_indices))
np.save(os.path.join(workspace, "end_tokens.npy"), np.array(end_indices))
