import os
import json
import numpy as np
from datetime import datetime
from PIL import Image
from sentence_transformers import SentenceTransformer
from evaluators.base_evaluator import BaseEvaluator
import config

class ClipEvaluator(BaseEvaluator):
    def __init__(self):
        # use CLIP model for embeddings
        print("Loading CLIP model for evaluation...")
        self.clip_model = SentenceTransformer('clip-ViT-L-14')

    def evaluate(self, sentences: list[str], matched_filenames: list[list[str]]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.OUTPUT_DIR, f"clip_scores_{timestamp}.json")
        
        # results contains sentence, image-filename and clip-similarity for the top-1 matched image
        results = []
        for sentence, top_k_list in zip(sentences, matched_filenames):
            filename = top_k_list[0]
            img_path = os.path.join(config.IMAGE_FOLDER, filename)
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_emb = self.clip_model.encode([img], convert_to_numpy=True)
                text_emb = self.clip_model.encode([sentence], convert_to_numpy=True)
                
                # Normalize and compute dot product
                img_emb /= np.linalg.norm(img_emb)
                text_emb /= np.linalg.norm(text_emb)
                score = float(np.dot(img_emb, text_emb.T)[0][0] * 100)
                
            except Exception as e:
                score = 0.0
                print(f"Error scoring {filename}: {e}")

            results.append({
                "sentence": sentence,
                "image": filename,
                "clip_similarity": round(score, 2)
            })

        avg_score = np.mean([r["clip_similarity"] for r in results]) if results else 0.0
        
        output_data = {
            "average_clip_similarity": round(float(avg_score), 2),
            "results": results
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
            
        print(f"Evaluation JSON saved to: {output_path} (Avg Score: {avg_score:.2f})")
