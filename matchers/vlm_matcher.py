import os
import json
import torch
import pickle
import re
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline
from matchers.base_matcher import BaseMatcher
import config


class VLMMatcher(BaseMatcher):
    def __init__(self):
        self.pkl_path = config.HYBRID_INDEX
        self.st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

        with open(self.pkl_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.w = {'cap': 0.48, 'act': 0.15, 'exe': 0.15, 'obj': 0.15, 'set': 0.02, 'em': 0.05}

        # model for intent extraction from input sentence
        print("Loading Qwen2.5-1.5B for intent extraction...")
        llm_model = 'Qwen/Qwen2.5-1.5B-Instruct'
        self.llm = pipeline("text-generation", model=llm_model, device_map="auto", torch_dtype=torch.float16)

        # VLM for re-ranking
        print("Loading Gemma 4 E4B for visual re-ranking...")
        self.vlm_id = "google/gemma-4-E4B-it"
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            self.vlm_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.vlm.eval()
        self.processor = AutoProcessor.from_pretrained(self.vlm_id)

    def _get_intent(self, query: str) -> dict:
        prompt = f"Extract actors, actions, objects (lists) and setting, emotion (string) from: '{query}'. Output strictly ONLY JSON."
        messages = [{"role": "user", "content": prompt}]
        fmt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        out = self.llm(fmt, max_new_tokens=150, do_sample=False, temperature=0.0)
        text = out[0]["generated_text"][len(fmt):].strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)

        try:
            return json.loads(match.group(0)) if match else {}
        except:
            return {}

    def _parse_decision(self, response_text: str, n_images: int) -> dict:
        """Parse the output of the model to get image, confidence score and winner"""
        clean = re.sub(r'```json|```', '', response_text).strip()

        # Look for the per-image scores array first
        scores_match = re.search(r'"scores"\s*:\s*\[(.*?)\]', clean, re.DOTALL)
        if scores_match:
            try:
                arr_str = '[' + scores_match.group(1) + ']'
                arr_str = re.sub(r',\s*([\]\}])', r'\1', arr_str)  # trailing commas
                scores = json.loads(arr_str)
                # Pick the index with the highest score
                best = max(range(len(scores)), key=lambda i: scores[i].get('score', 0))
                best_idx = scores[best].get('index', best)
                # Validate index
                if not (0 <= best_idx < n_images):
                    best_idx = best
                return {
                    'best_index': int(best_idx),
                    'confidence': float(scores[best].get('score', 0)),
                    'reason': scores[best].get('reason', ''),
                    'all_scores': scores,
                }
            except Exception as e:
                return {'_error': f'scores parse failed: {e}'}

        return {'_error': 'no scores array found'}

    def match(self, sentences: list[str], top_k: int = 1) -> list[list[str]]:
        results = []
        confidences = []
        all_scores_log = []
        CANDIDATE_POOL = 5

        def ensure_list(val):
            if isinstance(val, list):
                return [str(item) for item in val]
            if isinstance(val, str) and val.strip():
                return [val]
            return []

        SYSTEM_GUIDANCE = (
            "You are a strict Evaluator for AAC Pictograms. Your goal is literal accuracy.\n\n"
            "CRITICAL RULES:\n"
            "1. NO CONTRADICTIONS: If a sentence implies absence ('clear sky'), an image with the object ('cloud') is a FAILURE.\n"
            "2. LITERAL MATCHING: Prioritize images that lack distracting elements.\n"
            "3. SUBJECT FOCUS: Ensure the primary actor and action are the main focus.\n\n"
            "CONFIDENCE SCALE (use the FULL range, be discriminating):\n"
            "- 10: All key elements depicted, no contradictions, no distractors.\n"
            "- 8-9: All key elements depicted, very minor abstraction or stylization.\n"
            "- 6-7: Main subject and action correct, but missing a secondary element OR a small distractor present.\n"
            "- 4-5: Partial match: either the action OR the subject is right, but not both.\n"
            "- 2-3: Only loosely related (shared theme but wrong action/subject).\n"
            "- 1: Direct contradiction or completely unrelated.\n\n"
            "Do NOT default to extreme scores. Most real matches sit between 5 and 9. "
            "Reserve 10 for flawless matches and 1 for outright contradictions. "
            "When scoring multiple images, the scores SHOULD differ — if two images are equally good, "
            "look harder for a tie-breaker."
        )

        for query in sentences:
            # intent based matching with precomputed embeddings
            intent = self._get_intent(query)

            act_list = ensure_list(intent.get('actors', []))
            exe_list = ensure_list(intent.get('actions', []))
            obj_list = ensure_list(intent.get('objects', []))
            setting_str = str(intent.get('setting', '')).strip()
            emotion_str = str(intent.get('emotion', '')).strip()

            q_cap = self.st_model.encode(query, convert_to_tensor=True)
            q_act = self.st_model.encode(act_list, convert_to_tensor=True) if act_list else torch.empty((0, 384))
            q_exe = self.st_model.encode(exe_list, convert_to_tensor=True) if exe_list else torch.empty((0, 384))
            q_obj = self.st_model.encode(obj_list, convert_to_tensor=True) if obj_list else torch.empty((0, 384))
            q_set = self.st_model.encode(setting_str, convert_to_tensor=True) if setting_str else None
            q_em = self.st_model.encode(emotion_str, convert_to_tensor=True) if emotion_str else None

            scored_files = []
            for doc in self.dataset:
                score = self.w['cap'] * util.cos_sim(q_cap, doc['emb_caption']).item()

                for q_t, d_t, weight in [(q_act, doc.get('emb_actors'), self.w['act']),
                                         (q_exe, doc.get('emb_actions'), self.w['exe']),
                                         (q_obj, doc.get('emb_objects'), self.w['obj'])]:
                    if q_t.shape[0] > 0 and d_t is not None and d_t.shape[0] > 0:
                        sims = util.cos_sim(q_t, d_t)
                        score += weight * torch.mean(torch.max(sims, dim=1)[0]).item()

                if q_set is not None and doc.get('emb_setting') is not None:
                    score += self.w['set'] * util.cos_sim(q_set, doc['emb_setting']).item()
                if q_em is not None and doc.get('emb_emotion') is not None:
                    score += self.w['em'] * util.cos_sim(q_em, doc['emb_emotion']).item()

                scored_files.append((score, doc['original_filename']))

            scored_files.sort(key=lambda x: x[0], reverse=True)
            top_filenames = [os.path.basename(f) for _, f in scored_files[:CANDIDATE_POOL]]

            # top 5 filenames
            print(f"\nQuery: '{query}'")
            print("Retrieval pool (Stage 1):")
            for i, (s, fname) in enumerate(scored_files[:CANDIDATE_POOL]):
                print(f"  [{i}] {s:.4f}  {os.path.basename(fname)}")

            # second part with gemma 4 model
            images = []
            for f in top_filenames:
                path = os.path.join(config.IMAGE_FOLDER, f)
                images.append(Image.open(path).convert("RGB"))

            n = len(images)

            # add each image to the content of the prompt
            content = []
            for i, img in enumerate(images):
                content.append({"type": "text", "text": f"Image {i}:"})
                content.append({"type": "image", "image": img})

            instructions = (
                f"{SYSTEM_GUIDANCE}\n\n"
                f"Target Sentence: '{query}'\n\n"
                f"You have been shown {n} pictograms labelled Image 0 through Image {n-1}.\n\n"
                "Solve this in TWO STEPS. The images stay visible — keep looking at them.\n\n"
                "STEP 1 — DESCRIBE each image literally in 1-2 sentences. Cover:\n"
                "  - What entities/people/objects are visibly present\n"
                "  - What action (if any) is depicted\n"
                "  - Anything that CONTRADICTS the target sentence\n"
                "Be strictly literal. Do not infer meaning that is not visually present.\n\n"
                "STEP 2 — SCORE EVERY IMAGE on the 1-10 scale above. Do not skip any. "
                "Spread the scores: if two images feel similar, find a tie-breaker.\n\n"
                "OUTPUT FORMAT (exact):\n\n"
                "Descriptions:\n"
                "Image 0: <description>\n"
                "Image 1: <description>\n"
                f"...through Image {n-1}\n\n"
                "Then ONLY a JSON object on the last lines:\n"
                "{\n"
                '  "scores": [\n'
                '    {"index": 0, "score": X, "reason": "..."},\n'
                '    {"index": 1, "score": X, "reason": "..."},\n'
                f'    ...through index {n-1}\n'
                "  ]\n"
                "}"
            )
            content.append({"type": "text", "text": instructions})

            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.vlm.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                output = self.vlm.generate(
                    **inputs,
                    max_new_tokens=1200,
                    do_sample=False,
                )

            generated_ids = output[0][input_len:]
            response_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

            decision = self._parse_decision(response_text, n_images=n)

            if "_error" in decision:
                # choose image with best similarity if parsing error
                best_idx = 0
                conf = 5
                reason = f"[parse fallback] {decision['_error']}"
                all_scores_log.append(None)
            else:
                best_idx = decision['best_index']
                conf = decision['confidence']
                reason = decision['reason']
                all_scores_log.append(decision.get('all_scores'))

            # make sure index is in right range
            best_idx = min(max(int(best_idx), 0), n - 1)
            best_filename = top_filenames[best_idx]

            print(f"-> Selected: [{best_idx}] {best_filename} | Conf: {conf}/10 | Why: {str(reason)}")
            if decision.get('all_scores'):
                print(f"   Full scores: {[(s.get('index'), s.get('score')) for s in decision['all_scores']]}")

            results.append([best_filename])
            confidences.append(conf)

        return [results, confidences]
