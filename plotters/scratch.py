import json

with open('../common/pos_tags.json', 'r') as f:
    pos_tags = json.load(f)

for pos_id, pos_tag in enumerate(pos_tags):
    print(f"- {pos_id}: {pos_tag['tag']} ({pos_tag['title']})")
