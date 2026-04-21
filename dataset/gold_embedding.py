import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from get_image_embedding import get_image_embedding

# --- Configuration ---
GOLD_IMAGES_DIR = Path("easyread-retrieval-dataset/gold_standard_pic")
GOLD_METADATA_FILE = Path("easyread-retrieval-dataset/gold_standard.jsonl")
EMBEDDINGS_CACHE = Path(__file__).resolve().parent / "gold_embeddings.pkl"

gold_labels_map = {}

with open(GOLD_METADATA_FILE, "r") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON at line {line_no}: {e}")
            continue

        file_name = data.get("file_name")
        intent = data.get("intent")
        if not file_name or intent is None:
            print(f"Skipping line {line_no}: missing file_name or intent")
            continue

        gold_labels_map[Path(file_name).name] = intent

print(f"Loaded {len(gold_labels_map)} labels.")

def build_gold_library():
    gold_library_vectors = {}
    image_paths = list(GOLD_IMAGES_DIR.rglob("*.[jp][pn][g]"))
    print(f"Embedding {len(image_paths)} gold standard images...")

    for img_path in tqdm(image_paths):
        img_name = img_path.name
        if img_name in gold_labels_map:
            try:
                vector = get_image_embedding(str(img_path))
                gold_library_vectors[img_name] = {
                    "path": str(img_path),
                    "vector": vector,
                    "label": gold_labels_map[img_name]
                }
            except Exception as e:
                print(f"Error embedding {img_name}: {e}")

    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(gold_library_vectors, f)

    return gold_library_vectors

if os.path.exists(EMBEDDINGS_CACHE):
    with open(EMBEDDINGS_CACHE, "rb") as f:
        gold_library_vectors = pickle.load(f)
else:
    gold_library_vectors = build_gold_library()