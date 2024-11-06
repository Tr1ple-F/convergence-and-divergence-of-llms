import nltk
import json

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# Sample text
with open("common/input_text.txt", 'r', encoding="utf8") as file:
    text = file.read()

def align_pos_tags(hf_tokens, text):
    # Tokenize the text using NLTK
    nltk_tokens = nltk.word_tokenize(text)
    nltk_pos_tags = nltk.pos_tag(nltk_tokens)

    print("NLTK tokens:")
    print(len(nltk_tokens))
    print("HF tokens:")
    print(len(hf_tokens))

    # Initialize variables
    hf_index = 0
    aligned_tags = []

    # Iterate over NLTK tokens and align with Hugging Face tokens
    for nltk_token, pos_tag in nltk_pos_tags:
        # print(nltk_token)
        # print(pos_tag)
        current_token = ""
        while hf_index < len(hf_tokens) and current_token != nltk_token:
            current_token += hf_tokens[hf_index].replace("##", "")
            hf_index += 1
        aligned_tags.append((current_token, pos_tag))

    return aligned_tags


with open("common/input_text_tokenized.json", 'r') as file:
    hf_tokens = json.load(file)
    hf_tokens.insert(0, "<|endoftext|>")
    hf_tokens.append("<|endoftext|>")

aligned_tags = align_pos_tags(hf_tokens, text)
print(len(aligned_tags))
