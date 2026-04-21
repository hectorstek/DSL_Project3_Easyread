import json
import pickle
from sentence_transformers import SentenceTransformer
import config

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
dataset = []

# Read JSONL and embed the raw_caption
with open(config.METADATA_JSON, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line.strip())
        dataset.append({
            'file_name': obj['file_name'],
            'embedding': model.encode(obj['raw_caption'], convert_to_tensor=True)
        })

# Save to pickle
with open(config.CAPTION_INDEX, 'wb') as f:
    pickle.dump(dataset, f)

print("Precomputation complete.")
