import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import config
from matchers.base_matcher import BaseMatcher

class FilenameClipMatcher(BaseMatcher):
    def __init__(self):
        self.text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.clip_model = SentenceTransformer('clip-ViT-L-14')
        data = np.load(config.FILENAME_INDEX_FILE)
        self.filename_vectors, self.filenames = data['embeddings'], data['files']

    def match(self, sentences, top_k=10):
        return [self._find_best(s, top_k) for s in sentences]

    # candidate_k is number of items to consider after text search
    # final_k is number of items to return
    def _find_best(self, query, final_k, candidate_k=20):
        # Step 1: Text search only based on filename
        query_text_vec = self.text_model.encode([query], convert_to_numpy=True)
        text_sims = np.dot(self.filename_vectors, query_text_vec.T).flatten()
        candidate_files = [self.filenames[i] for i in text_sims.argsort()[-candidate_k:][::-1]]

        # Step 2: ranking candidate_k images based on CLIP
        images, valid_files = [], []
        for f in candidate_files:
            try:
                images.append(Image.open(os.path.join(config.IMAGE_FOLDER, f)).convert("RGB"))
                valid_files.append(f)
            except: continue

        img_embs = self.clip_model.encode(images, convert_to_numpy=True)
        query_clip_vec = self.clip_model.encode([query], convert_to_numpy=True)

        # Normalize and compute similarity
        img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True)
        query_clip_vec /= np.linalg.norm(query_clip_vec)
        visual_sims = np.dot(img_embs, query_clip_vec.T).flatten()

        # Get top-k filenames
        best_indices = visual_sims.argsort()[-final_k:][::-1]
        return [valid_files[i] for i in best_indices]




















'''


import os
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from matchers.base_matcher import BaseMatcher
import config

class FilenameClipMatcher(BaseMatcher):
    def __init__(self):
        self.text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.clip_model = SentenceTransformer('clip-ViT-L-14')

        if not os.path.exists(config.INDEX_FILE):
            raise FileNotFoundError(f"Missing index file: {config.INDEX_FILE}")

        data = np.load(config.INDEX_FILE)
        self.filename_vectors = data['embeddings']
        self.filenames = data['files']

    def match(self, sentences: list[str]) -> list[str]:
        matched_files = []
        for i, sentence in enumerate(sentences):
            print(f"Matching ({i+1}/{len(sentences)}): {sentence}")
            best_file = self._find_best_pictogram(sentence)
            matched_files.append(best_file)
        return matched_files

    def _find_best_pictogram(self, query: str, top_k: int = 20) -> str:
        # Step 1: Filename retrieval
        query_text_vec = self.text_model.encode([query], convert_to_numpy=True)
        text_sims = np.dot(self.filename_vectors, query_text_vec.T).flatten()
        top_k_indices = text_sims.argsort()[-top_k:][::-1]
        candidate_files = [self.filenames[i] for i in top_k_indices]

        # Step 2: Visual Re-ranking
        images, valid_files = [], []
        for f in candidate_files:
            try:
                path = os.path.join(config.IMAGE_FOLDER, f)
                images.append(Image.open(path).convert("RGB"))
                valid_files.append(f)
            except Exception:
                continue

        img_embs = self.clip_model.encode(images, convert_to_numpy=True)
        query_clip_vec = self.clip_model.encode([query], convert_to_numpy=True)

        img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True)
        query_clip_vec /= np.linalg.norm(query_clip_vec)

        visual_sims = np.dot(img_embs, query_clip_vec.T).flatten()
        winner_idx = np.argmax(visual_sims)

        return valid_files[winner_idx]
'''
