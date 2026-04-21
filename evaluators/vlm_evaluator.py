import os
import json
import torch
from datetime import datetime
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import config
from evaluators.base_evaluator import BaseEvaluator
import re

class VLMEvaluator(BaseEvaluator):
    def __init__(self):
        print("Loading LLaVA-v1.6-7B (Reasoning Heavyweight)...")
        self.model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

        # Optimized for your 64GB RAM / CPU setup
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32, # CPU needs float32 or bfloat16
            device_map="cpu",          # Explicitly use your strong CPU
            low_cpu_mem_usage=True
        )

    def extract_score(self, text):
        # extract number from text output
        match = re.search(r"Score:\s*(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def evaluate(self, sentences, matched_filenames):
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        for sentence, filenames in zip(sentences, matched_filenames):
            img_path = os.path.join(config.IMAGE_FOLDER, filenames[0])
            image = Image.open(img_path).convert("RGB")


            SYSTEM_GUIDANCE = (
                "You are a critical evaluator for 'Easy Read' pictograms. Use the full 1-10 scale.\n"
                "Example 1: Text 'Eating' -> Image is just an apple. Score: 5/10 (Object shown, but action is missing).\n"
                "Example 2: Text 'Running' -> Image is a person running. Score: 10/10 (Perfect literal match).\n"
                "Example 3: Text 'Sunny' -> Image is a rain cloud. Score: 1/10 (Incorrect).\n"
                "Aim for a realistic distribution: most good matches are 7-8, only perfect ones are 10."
            )

            prompt = (
                f"[INST] <image>\n{SYSTEM_GUIDANCE}\n"
                f"Sentence: '{sentence}'\n"
                "Provide your evaluation strictly in this format:\n"
                "Score: [X]/10\n"
                "Reason: [Short explanation] [/INST]"
            )

            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cpu")
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True
            )
            response = self.processor.decode(output[0], skip_special_tokens=True)

            # remove the prompt from the response
            clean_response = response.split("[/INST]")[-1].strip()

            results.append({
                "sentence": sentence,
                "image": filenames[0],
                "score": self.extract_score(clean_response),
                "llava_reasoning": clean_response
            })
            print(f"Evaluated: {filenames[0]} -> {clean_response[:60]}...")

        out_file = os.path.join(config.OUTPUT_DIR, f"llava_reasoning_{timestamp}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Llava Evaluation saved to {out_file}")
