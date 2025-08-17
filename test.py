import json

# Load your current knowledge base
with open("knowledge_base.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare new data structure with IDs
new_data = {}

for category, texts in data.items():
    new_data[category] = []
    for i, text in enumerate(texts, start=1):
        entry = {
            "id": f"{category[:3].upper()}-{i}",  # Example: PHI-1, PHI-2, etc.
            "text": text
        }
        new_data[category].append(entry)

# Save the new JSON
with open("knowledge_base_with_ids.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("✅ New knowledge base created with manual IDs!")
