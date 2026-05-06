'''
import json
import pickle
import re
import torch
import os
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from matchers.base_matcher import BaseMatcher
import config

class LLMMatcher(BaseMatcher):
    def __init__(self, llm_model: str = 'Qwen/Qwen2.5-1.5B-Instruct'):
        self.pkl_path = config.HYBRID_INDEX
        self.st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.llm = pipeline("text-generation", model=llm_model, device_map="auto", torch_dtype=torch.float16)

        with open(self.pkl_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.default_w = {'cap': 5, 'act': 2, 'exe': 2, 'obj': 2, 'set': 1, 'em': 2}

    def _get_intent(self, query: str) -> dict:
        """
        One single LLM call to extract both the components and the importance weights.
        """
        prompt = f"""
        Extract actors, actions, objects (lists), setting, and emotion (string) from: '{query}'.
        Also provide an 'importance' dictionary scoring each category from 1 to 5 based on its focus.
        Output strictly ONLY JSON in this format:
        {{"actors": [], "actions": [], "objects": [], "setting": "", "emotion": "", "importance": {{"cap": 5, "act": 1, "exe": 1, "obj": 1, "set": 1, "em": 1}}}}
        """
        messages = [{"role": "user", "content": prompt}]
        fmt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        out = self.llm(fmt, max_new_tokens=150, do_sample=False, temperature=0.0)
        text = out[0]["generated_text"][len(fmt):].strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)

        try:
            return json.loads(match.group(0)) if match else {}
        except:
            return {}

    def match(self, sentences: list[str], top_k: int = 1) -> list[list[str]]:
        results = []

        def ensure_list(val):
            if isinstance(val, list): return [str(item) for item in val]
            if isinstance(val, str) and val.strip(): return [val]
            return []

        for query in sentences:
            intent = self._get_intent(query)

            # --- DYNAMIC RE-WEIGHTING ---
            raw_weights = intent.get('importance', self.default_w)
            total_w = sum(raw_weights.values()) or 1
            dyn_w = {k: v / total_w for k, v in raw_weights.items()}

            # --- PREPARE QUERY EMBEDDINGS ---
            q_cap = self.st_model.encode(query, convert_to_tensor=True)

            # Using lists for multi-word fragments (Actors, Actions, Objects)
            act_l = ensure_list(intent.get('actors', []))
            exe_l = ensure_list(intent.get('actions', []))
            obj_l = ensure_list(intent.get('objects', []))

            q_act = self.st_model.encode(act_l, convert_to_tensor=True) if act_l else None
            q_exe = self.st_model.encode(exe_l, convert_to_tensor=True) if exe_l else None
            q_obj = self.st_model.encode(obj_l, convert_to_tensor=True) if obj_l else None

            # Using strings for Setting/Emotion
            set_s = str(intent.get('setting', '')).strip()
            emo_s = str(intent.get('emotion', '')).strip()
            q_set = self.st_model.encode(set_s, convert_to_tensor=True) if set_s else None
            q_em = self.st_model.encode(emo_s, convert_to_tensor=True) if emo_s else None

            scored_files = []
            for doc in self.dataset:
                # 1. Base Caption Score
                score = dyn_w.get('cap', 0.4) * util.cos_sim(q_cap, doc['emb_caption']).item()

                # 2. Fragment Scores (Max-Mean Pooling for lists)
                for q_vec, d_key, w_key in [(q_act, 'emb_actors', 'act'),
                                           (q_exe, 'emb_actions', 'exe'),
                                           (q_obj, 'emb_objects', 'obj')]:
                    d_vec = doc.get(d_key)
                    if q_vec is not None and d_vec is not None and d_vec.shape[0] > 0:
                        sims = util.cos_sim(q_vec, d_vec)
                        score += dyn_w.get(w_key, 0.1) * torch.mean(torch.max(sims, dim=1)[0]).item()

                # 3. Single-Vector Scores (Setting & Emotion)
                if q_set is not None and doc.get('emb_setting') is not None:
                    score += dyn_w.get('set', 0.05) * util.cos_sim(q_set, doc['emb_setting']).item()

                if q_em is not None and doc.get('emb_emotion') is not None:
                    score += dyn_w.get('em', 0.1) * util.cos_sim(q_em, doc['emb_emotion']).item()

                scored_files.append((score, doc['original_filename']))

            # --- SORT AND RETURN ---
            scored_files.sort(key=lambda x: x[0], reverse=True)

            final_top = []
            for _, full_path in scored_files[:top_k]:
                filename = os.path.basename(full_path)
                if not filename.lower().endswith('.png'):
                    filename = f"{os.path.splitext(filename)[0]}.png"
                final_top.append(filename)

            results.append(final_top)

        return results

'''
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

class LLMMatcher(BaseMatcher):
    def __init__(self, llm_model: str = 'Qwen/Qwen2.5-1.5B-Instruct'):
        self.pkl_path = config.HYBRID_INDEX

        # 1. Vector Search Engine
        self.st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 2. Semantic Reasoner / Re-ranker
        self.llm = pipeline("text-generation", model=llm_model, device_map="auto", torch_dtype=torch.float16)

        with open(self.pkl_path, 'rb') as f:
            self.dataset = pickle.load(f)

        # Fallback weights in case the LLM fails to generate the importance dictionary
        self.default_w = {'cap': 5, 'act': 2, 'exe': 2, 'obj': 2, 'set': 1, 'em': 1}

    def _get_intent(self, query: str) -> dict:
        """
        Stage 1: Extract components and dynamically ask the LLM for category weights.
        """
        prompt = f"""
        Extract actors, actions, objects (lists), setting, and emotion (string) from the sentence: '{query}'.
        Also provide an 'importance' dictionary scoring each category from 1 to 5 based on its focus in the sentence.
        Categories are: 'cap' (overall caption), 'act' (actors), 'exe' (actions), 'obj' (objects), 'set' (setting), 'em' (emotion).
        Output strictly ONLY JSON in this exact format:
        {{"actors": [], "actions": [], "objects": [], "setting": "", "emotion": "", "importance": {{"cap": 5, "act": 1, "exe": 1, "obj": 1, "set": 1, "em": 1}}}}
        """
        messages = [{"role": "user", "content": prompt}]
        fmt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        out = self.llm(fmt, max_new_tokens=200, do_sample=False, temperature=0.0)
        text = out[0]["generated_text"][len(fmt):].strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)

        try:
            return json.loads(match.group(0)) if match else {}
        except:
            return {}
    def _rerank_candidates(self, query: str, intent: dict, top_docs: list) -> int:
        """
        Stage 2: Pass the top N vector matches WITH ALL METADATA (including setting) to the LLM.
        """
        if not top_docs:
            return 0

        # Create a detailed "Context Profile" for each candidate
        candidates_text = ""
        for i, doc in enumerate(top_docs):
            cap = doc.get('caption', 'No caption')
            act = ", ".join(doc.get('actors', [])) or "None"
            exe = ", ".join(doc.get('actions', [])) or "None"
            obj = ", ".join(doc.get('objects', [])) or "None"
            set_ = doc.get('setting', 'General/None') # Added Setting
            emo = doc.get('emotion', 'Neutral')

            candidates_text += (
                f"ID [{i}]:\n"
                f"  - Caption: {cap}\n"
                f"  - Actors: {act}\n"
                f"  - Actions: {exe}\n"
                f"  - Objects: {obj}\n"
                f"  - Setting: {set_}\n"
                f"  - Emotion: {emo}\n\n"
            )

        prompt = f"""
        User Query: '{query}'
        Desired Intent: {json.dumps(intent)}

        You are an expert at matching sentences to visual symbols.
        Compare the Desired Intent above against the {len(top_docs)} candidates below.
        Pay close attention to the SETTING (where it happens) and ACTIONS.

        CANDIDATES:
        {candidates_text}

        Which ID is the most accurate match for the query?
        Return ONLY the ID number.
        """

        messages = [{"role": "user", "content": prompt}]
        fmt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Qwen-1.5B is quite fast, so we keep max_new_tokens low to ensure a snappy response
        out = self.llm(fmt, max_new_tokens=10, do_sample=False, temperature=0.0)
        text = out[0]["generated_text"][len(fmt):].strip()

        # Regex to find the ID number in the response
        num_match = re.search(r'\d+', text)
        if num_match:
            idx = int(num_match.group())
            if 0 <= idx < len(top_docs):
                return idx

        return 0 # Fallback to first vector result

    def match(self, sentences: list[str], top_k: int = 1) -> list[list[str]]:
        results = []

        def ensure_list(val):
            if isinstance(val, list):
                return [str(item) for item in val]
            if isinstance(val, str) and val.strip():
                return [val]
            return []

        # We retrieve more documents from the vector DB so the LLM has options
        RETRIEVAL_POOL_SIZE = 15

        for query in sentences:
            intent = self._get_intent(query)

            # --- DYNAMIC RE-WEIGHTING LOGIC ---
            raw_weights = intent.get('importance', self.default_w)

            # Ensure all keys exist to prevent KeyError
            for k in self.default_w.keys():
                if k not in raw_weights:
                    raw_weights[k] = self.default_w[k]

            total_w = sum(raw_weights.values())
            if total_w == 0: total_w = 1 # Prevent division by zero

            # Normalize so they sum to 1.0
            dyn_w = {k: v / total_w for k, v in raw_weights.items()}
            # ----------------------------------

            # Sanitize inputs
            act_list = ensure_list(intent.get('actors', []))
            exe_list = ensure_list(intent.get('actions', []))
            obj_list = ensure_list(intent.get('objects', []))
            setting_str = str(intent.get('setting', '')).strip()
            emotion_str = str(intent.get('emotion', '')).strip()

            # Embed Query Components
            q_cap = self.st_model.encode(query, convert_to_tensor=True)
            q_act = self.st_model.encode(act_list, convert_to_tensor=True) if act_list else torch.empty((0, 384))
            q_exe = self.st_model.encode(exe_list, convert_to_tensor=True) if exe_list else torch.empty((0, 384))
            q_obj = self.st_model.encode(obj_list, convert_to_tensor=True) if obj_list else torch.empty((0, 384))
            q_set = self.st_model.encode(setting_str, convert_to_tensor=True) if setting_str else None
            q_em = self.st_model.encode(emotion_str, convert_to_tensor=True) if emotion_str else None

            scored_files = []
            for doc in self.dataset:
                # 1. Caption Score
                score = dyn_w['cap'] * util.cos_sim(q_cap, doc['emb_caption']).item()

                # 2. Intent Scores (Max-Mean Pooling)
                for q_t, d_t, weight in [(q_act, doc.get('emb_actors'), dyn_w['act']),
                                         (q_exe, doc.get('emb_actions'), dyn_w['exe']),
                                         (q_obj, doc.get('emb_objects'), dyn_w['obj'])]:
                    if q_t.shape[0] > 0 and d_t is not None and d_t.shape[0] > 0:
                        sims = util.cos_sim(q_t, d_t)
                        score += weight * torch.mean(torch.max(sims, dim=1)[0]).item()

                # 3. Setting Score
                if q_set is not None and doc.get('emb_setting') is not None:
                    score += dyn_w['set'] * util.cos_sim(q_set, doc['emb_setting']).item()

                # 4. Emotion Score (Fixed!)
                if q_em is not None and doc.get('emb_emotion') is not None:
                    score += dyn_w['em'] * util.cos_sim(q_em, doc['emb_emotion']).item()

                scored_files.append((score, doc))

            # Sort by vector similarity score
            scored_files.sort(key=lambda x: x[0], reverse=True)

            # Extract the top N actual document dictionaries for the Re-ranker
            top_candidate_docs = [doc for score, doc in scored_files[:RETRIEVAL_POOL_SIZE]]

            # --- LLM RE-RANKING LOGIC ---
            best_idx = self._rerank_candidates(query, intent, top_candidate_docs)

            # Move the LLM's chosen document to the very front of the list
            chosen_doc = top_candidate_docs.pop(best_idx)
            top_candidate_docs.insert(0, chosen_doc)
            # ----------------------------

            # Format final output
            final_top = []
            for doc in top_candidate_docs[:top_k]:
                filename = os.path.basename(doc['original_filename'])
                if not filename.lower().endswith('.png'):
                    filename = f"{os.path.splitext(filename)}.png"
                final_top.append(filename)

            results.append(final_top)

        return results

