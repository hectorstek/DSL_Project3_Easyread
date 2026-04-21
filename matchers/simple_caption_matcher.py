import pickle
import torch
import os
from sentence_transformers import SentenceTransformer, util
from matchers.base_matcher import BaseMatcher

class SimpleCaptionMatcher(BaseMatcher):
    def __init__(self, pkl_path: str, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)
            
        # Extract filenames and stack all embeddings into a single tensor for lightning-fast matching
        self.file_names = [item['file_name'] for item in dataset]
        self.doc_embeddings = torch.stack([item['embedding'] for item in dataset])

    def match(self, sentences: list[str], top_k: int = 1) -> list[list[str]]:
        # Encode all input sentences at once
        query_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # Compute cosine similarity: matrix of shape [num_queries, num_docs]
        cos_scores = util.cos_sim(query_embeddings, self.doc_embeddings)
        
        all_results = []
        for i in range(len(sentences)):
            # Find the top_k highest scores for query 'i'
            top_results = torch.topk(cos_scores[i], k=top_k)
            
            clean_filenames = []
            for idx in top_results.indices:
                raw_name = self.file_names[idx]
                base = os.path.basename(raw_name)
                name_without_ext = os.path.splitext(base)[0]
                clean_filenames.append(f"{name_without_ext}.png")
                
            all_results.append(clean_filenames)
            
        return all_results
'''
# --- Usage Example ---
if __name__ == "__main__":
    matcher = SimpleCaptionMatcher("/home/linus/easyread/Linus/Modular/caption_embeddings.pkl")
    results = matcher.match(
        sentences=["A person pointing at a list of numbers.", "The national flag of Lithuania."], 
        top_k=2
    )
    print(results)
'''
