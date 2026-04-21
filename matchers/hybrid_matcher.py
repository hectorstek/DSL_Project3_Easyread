import json
import pickle
import re
import torch
import os
import pymupdf as fitz
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from matchers.base_matcher import BaseMatcher
import config

class HybridMatcher(BaseMatcher):
    def __init__(self, llm_model: str = 'Qwen/Qwen2.5-1.5B-Instruct'):
        # use multilingual model with 384 dim
        self.pkl_path = config.HYBRID_INDEX
        self.st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.llm = pipeline("text-generation", model=llm_model, device_map="auto", torch_dtype=torch.float16)

        with open(self.pkl_path, 'rb') as f:
            #self.dataset = torch.load(f, map_location="cpu") #pickle.load(f)
            #self.dataset = torch.load(f, map_location="cpu", weights_only=False)
            self.dataset = pickle.load(f)

        self.w = {'cap': 0.50, 'act': 0.15, 'exe': 0.20, 'obj': 0.10, 'set': 0.05}

    # create intent for input sentence
    def _get_intent(self, query: str) -> dict:
        prompt = f"Extract actors, actions, objects (lists) and setting (string) from: '{query}'. Output strictly ONLY JSON."
        messages = [{"role": "user", "content": prompt}]
        fmt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # remove max_length to remove warning
        out = self.llm(fmt, max_new_tokens=150, do_sample=False, temperature=0.0)

        text = out[0]["generated_text"][len(fmt):].strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)

        try:
            return json.loads(match.group(0)) if match else {}
        except:
            return {}

    def match(self, sentences: list[str], top_k: int = 1) -> list[list[str]]:
        results = []

        # make sure to return list of strings
        def ensure_list(val):
            if isinstance(val, list):
                return [str(item) for item in val]
            if isinstance(val, str) and val.strip():
                return [val]
            return []

        for query in sentences:
            intent = self._get_intent(query)

            # Sanitize inputs from LLM
            act_list = ensure_list(intent.get('actors', []))
            exe_list = ensure_list(intent.get('actions', []))
            obj_list = ensure_list(intent.get('objects', []))
            setting_str = str(intent.get('setting', '')).strip()

            # Embed Query Components
            q_cap = self.st_model.encode(query, convert_to_tensor=True)

            # Use empty tensors if the list is empty to prevent SentenceTransformer crashes
            q_act = self.st_model.encode(act_list, convert_to_tensor=True) if act_list else torch.empty((0, 384))
            q_exe = self.st_model.encode(exe_list, convert_to_tensor=True) if exe_list else torch.empty((0, 384))
            q_obj = self.st_model.encode(obj_list, convert_to_tensor=True) if obj_list else torch.empty((0, 384))
            q_set = self.st_model.encode(setting_str, convert_to_tensor=True) if setting_str else None

            scored_files = []
            for doc in self.dataset:
                # compute similarity of caption score
                score = self.w['cap'] * util.cos_sim(q_cap, doc['emb_caption']).item()

                # intent scores (Max-Mean Pooling)
                # of each item in the input sentence, compute similarity with each item in dataset and take max. Then average over all items from input sentence.
                for q_t, d_t, weight in [(q_act, doc.get('emb_actors'), self.w['act']),
                                         (q_exe, doc.get('emb_actions'), self.w['exe']),
                                         (q_obj, doc.get('emb_objects'), self.w['obj'])]:

                    if q_t.shape[0] > 0 and d_t is not None and d_t.shape[0] > 0:
                        sims = util.cos_sim(q_t, d_t)
                        # We take the best match for each query fragment and average them
                        score += weight * torch.mean(torch.max(sims, dim=1)[0]).item()

                # compute similarity of setting (if it exists)
                if q_set is not None and doc.get('emb_setting') is not None:
                    score += self.w['set'] * util.cos_sim(q_set, doc['emb_setting']).item()

                scored_files.append((score, doc['original_filename']))

            # sort the files
            scored_files.sort(key=lambda x: x[0], reverse=True)

            # return output as filename
            final_top = []
            for _, full_path in scored_files[:top_k]:
                filename = os.path.basename(full_path)
                if not filename.lower().endswith('.png'):
                    filename = f"{os.path.splitext(filename)[0]}.png"
                final_top.append(filename)

            results.append(final_top)

        return results
