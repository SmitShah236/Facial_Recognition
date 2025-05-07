import json

# Path to your embeddings JSON file
json_path = 'Media/Embeddings.json'

# Load the JSON data
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fix paths
updated = False
for entry in data:
    if 'path' in entry and '\\' in entry['path']:
        entry['path'] = entry['path'].replace('\\', '/')
        updated = True

# Save the updated data
if updated:
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print("Updated paths with forward slashes.")
else:
    print("No paths needed updating.")
