import json
import pickle
import re
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from matchers.base_matcher import BaseMatcher
import config
import os

class JsonMatcher(BaseMatcher):
    # does not consider raw_caption
    def __init__(
        self,
        pkl_path: str,
        st_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
        llm_model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct',
        top_k: int = 1
    ):
        self.pkl_path = config.INTENT_INDEX
        self.st_model = SentenceTransformer(st_model_name)
        self.llm = pipeline("text-generation", model=llm_model_name, device_map="auto", torch_dtype=torch.float16)

        with open(pkl_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.top_k = top_k
        self.weights = {'actors': 0.40, 'actions': 0.40, 'objects': 0.15, 'setting': 0.05}

    def _ask_llm(self, prompt: str) -> str:
        """Minimal helper to query the LLM and extract the JSON block."""
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self.llm(formatted_prompt, max_new_tokens=200, do_sample=False, temperature=0.0)
        text = out[0]["generated_text"][len(formatted_prompt):].strip()

        # Extract anything that looks like JSON (dict or list)
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        return match.group(0) if match else text

    def _calculate_embedding_score(self, q_embs: dict, doc_embs: dict) -> float:
        total_score = 0.0
        for key in ['actors', 'actions', 'objects']:
            if not len(q_embs[key]) or not len(doc_embs[key]):
                continue
            cos_scores = util.cos_sim(q_embs[key], doc_embs[key])
            total_score += self.weights[key] * torch.mean(torch.max(cos_scores, dim=1)[0]).item()

        if q_embs['setting'] is not None and doc_embs['setting'] is not None:
            total_score += self.weights['setting'] * util.cos_sim(q_embs['setting'], doc_embs['setting']).item()

        return total_score

    def match(self, sentences: list[str], top_k: int = 1) -> list[list[str]]:
        all_results = []

        for query in sentences:
            # 1. Parse Query Intent
            intent_prompt = f"Extract actors, actions, objects (as lists) and setting (string) from: '{query}'. Output strictly ONLY JSON."
            try:
                query_intent = json.loads(self._ask_llm(intent_prompt))
            except json.JSONDecodeError:
                query_intent = {"actors": [], "actions": [], "objects": [], "setting": "unknown"}

            # 2. Embed Query
            q_embs = {
                k: self.st_model.encode(query_intent.get(k, []), convert_to_tensor=True) if query_intent.get(k) else []
                for k in ['actors', 'actions', 'objects']
            }
            q_embs['setting'] = self.st_model.encode(query_intent.get('setting', ''), convert_to_tensor=True) if query_intent.get('setting') else None

            # 3. Score all documents using Embeddings
            scores = []
            for doc in self.dataset:
                score = self._calculate_embedding_score(q_embs, doc['embeddings'])
                scores.append((score, doc))

            # 4. Get Top 20 Candidates
            scores.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [doc for _, doc in scores[:self.top_n]]

            # 5. LLM Reranking
            candidates_str = "\n".join([f"File: '{d['file_name']}' | Intent: {d['intent']}" for d in top_candidates])
            rerank_prompt = (
                f"Query: '{query}'.\n"
                f"Select the top {top_k} best matching files from these candidates:\n{candidates_str}\n"
                f"Output strictly ONLY a JSON list of strings containing the file names."
            )

            try:
                raw_filenames = json.loads(self._ask_llm(rerank_prompt))

                # FIX: Sanitize the LLM output to guarantee "filename.png"
                clean_filenames = []
                for f in raw_filenames[:top_k]:
                    base = os.path.basename(f) # Strips away any folder paths
                    name_without_ext = os.path.splitext(base)[0] # Strips any existing extension
                    clean_filenames.append(f"{name_without_ext}.png") # Forces .png

                all_results.append(clean_filenames)

            except json.JSONDecodeError:
                # FIX: Apply the same sanitization to the fallback
                fallback_filenames = []
                for d in top_candidates[:top_k]:
                    base = os.path.basename(d['file_name'])
                    name_without_ext = os.path.splitext(base)[0]
                    fallback_filenames.append(f"{name_without_ext}.png")

                all_results.append(fallback_filenames)

        return all_results

