import json
import os

json_path = 'Media\Embeddings.json'
base_dir = 'Media'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

missing = []

for entry in data:
    if 'path' in entry:
        path = os.path.join(base_dir, entry['path'])
        if not os.path.exists(path):
            missing.append(path)

if missing:
    print("Missing files:")
    for path in missing:
        print("-", path)
else:
    print("All files exist.")
