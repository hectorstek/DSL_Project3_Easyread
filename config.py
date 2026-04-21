import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_DIR = "/home/linus/DSL/ARASAAC_DE_unnest" #os.path.join(BASE_DIR, "arasaac_images_globalsymbols")
IMAGE_FOLDER = "/home/linus/DSL/ARASAAC_DE_unnest" #os.path.join(BASE_DIR, ) #"/Users/linus/Code/DSL/ARASAAC_DE_unnest"
#IMAGE_FOLDER = "/home/linus/easyread/Hector/easyread-retrieval-dataset/gold_standard_pic"
FILENAME_INDEX = os.path.join(BASE_DIR, "input", "arasaac_text_index.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GROUND_TRUTH_FILE = os.path.join(INPUT_DATA_DIR, "ground_truth_topk.json")
INTENT_INDEX = os.path.join(BASE_DIR, "input", "arasaac_text_index.npz")
METADATA_JSON = os.path.join(BASE_DIR, "input", "metadata_v2.1_12k.jsonl")
CAPTION_INDEX = os.path.join(BASE_DIR, "input", "caption_embeddings.pkl")
HYBRID_INDEX = os.path.join(BASE_DIR, "input", "hybrid_german_v2_12k.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)
