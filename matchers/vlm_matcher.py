import os
import json
import torch
import pickle
import re
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from matchers.base_matcher import BaseMatcher
import config

class VLMMatcher(BaseMatcher):
    def __init__(self):
        self.st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        with open(config.HYBRID_INDEX, 'rb') as f:
            self.dataset = pickle.load(f)

        print("Loading Qwen2.5-VL-7B (Expert Configuration)...")
        self.vlm_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vlm_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(self.vlm_id)

    def match(self, sentences: list[str], top_k: int = 1) -> list[list[str]]:
        results = []
        confidences = []
        CANDIDATE_POOL = 5

        # --- THE EXPERT PROMPT ---
        SYSTEM_GUIDANCE = (
            "You are a strict Evaluator for AAC Pictograms. Your goal is literal accuracy.\n\n"
            "CRITICAL RULES:\n"
            "1. NO CONTRADICTIONS: If a sentence says 'clear sky', an image with a cloud is a FAILURE. "
            "If a sentence says 'no water', an image with water is a FAILURE.\n"
            "2. LITERAL MATCHING: The image must represent the EXACT state of the sentence. "
            "Prioritize images that lack distracting elements.\n"
            "3. SUBJECT FOCUS: Ensure the primary actor and action are the main focus of the image.\n\n"
            "SCORING:\n"
            "- 10: Perfect match, no contradictions.\n"
            "- 1: Contradicts the sentence (e.g., has objects that shouldn't be there)."
        )

        for query in sentences:
            # Stage 1: Vector Search
            q_emb = self.st_model.encode(query, convert_to_tensor=True)
            scored_files = []
            for doc in self.dataset:
                score = util.cos_sim(q_emb, doc['emb_caption']).item()
                scored_files.append((score, doc['original_filename']))

            scored_files.sort(key=lambda x: x[0], reverse=True)
            top_filenames = [os.path.basename(f) for _, f in scored_files[:CANDIDATE_POOL]]

            # Stage 2: Visual Expert Re-ranking
            images = [Image.open(os.path.join(config.IMAGE_FOLDER, f)).convert("RGB") for f in top_filenames]
            print(images)
            image_placeholders = [{"type": "image"} for _ in range(len(images))]

            user_query = {
                "type": "text",
                "text": (
                    f"{SYSTEM_GUIDANCE}\n\n"
                    f"Target Sentence: '{query}'\n"
                    f"Evaluate the {len(images)} provided images. "
                    "Which index (0-{len(images)-1}) is the best match? "
                    "Respond ONLY with a JSON object: {\"best_index\": N, \"confidence\": X, \"reason\": \"...\"}"
                )
            }

            messages = [{"role": "user", "content": image_placeholders + [user_query]}]
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_input], images=images, padding=True, return_tensors="pt").to("cpu")

            # Use a slightly higher max_new_tokens to allow for the 'reason' field
            output = self.vlm.generate(**inputs, max_new_tokens=100, do_sample=False)

            generated_ids = output[0][len(inputs.input_ids[0]):]
            response_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

            try:
                # Find JSON block
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                decision = json.loads(match.group(0))
                best_idx = int(decision.get("best_index", 0))
                conf = decision.get("confidence", 0)
                reason = decision.get("reason", "No reason provided")
            except:
                best_idx, conf, reason = 0, 0, "Parsing failed"

            best_filename = top_filenames[min(max(best_idx, 0), len(top_filenames)-1)]
            print(f"-> Selected: {best_filename} with index {best_idx} | Conf: {conf}/10 | Why: {reason[:60]}...")

            results.append([best_filename])
            confidences.append(conf)

        return [results, confidences]
