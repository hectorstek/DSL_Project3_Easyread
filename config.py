import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#INPUT_DATA_DIR = "/home/linus/DSL/ARASAAC_DE_unnest" #os.path.join(BASE_DIR, "arasaac_images_globalsymbols")
#IMAGE_FOLDER = "/home/linus/DSL/ARASAAC_DE_unnest" #os.path.join(BASE_DIR, ) #"/Users/linus/Code/DSL/ARASAAC_DE_unnest"
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "easyread-retrieval-dataset", "data")
FILENAME_INDEX = os.path.join(BASE_DIR, "input", "arasaac_text_index.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
#GROUND_TRUTH_FILE = os.path.join(INPUT_DATA_DIR, "ground_truth_topk.json")
INTENT_INDEX = os.path.join(BASE_DIR, "input", "arasaac_text_index.npz")
METADATA_JSON = os.path.join(BASE_DIR, "dataset", "easyread-retrieval-dataset/metadata_v5.jsonl")
CAPTION_INDEX = os.path.join(BASE_DIR, "input", "caption_embeddings.pkl")
HYBRID_INDEX = os.path.join(BASE_DIR, "input", "index_v5.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)
