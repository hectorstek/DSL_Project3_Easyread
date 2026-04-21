import json
import pickle
import torch
from sentence_transformers import SentenceTransformer
import config

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
dataset = []

with open(config.METADATA_JSON, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line.strip())
        intent = obj.get('intent', {})

        # We store the EXACT file_name from the JSONL
        dataset.append({
            'original_filename': obj['file_name'],
            'emb_caption': model.encode(obj['raw_caption'], convert_to_tensor=True),
            'emb_actors': model.encode(intent.get('actors', []), convert_to_tensor=True) if intent.get('actors') else torch.empty((0, 384)),
            'emb_actions': model.encode(intent.get('actions', []), convert_to_tensor=True) if intent.get('actions') else torch.empty((0, 384)),
            'emb_objects': model.encode(intent.get('objects', []), convert_to_tensor=True) if intent.get('objects') else torch.empty((0, 384)),
            'emb_setting': model.encode(intent.get('setting', ''), convert_to_tensor=True) if intent.get('setting') else None
        })

with open(config.HYBRID_INDEX, 'wb') as f:
    pickle.dump(dataset, f)
