import os
import json
import torch
import re
from datetime import datetime
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import config
from evaluators.base_evaluator import BaseEvaluator

class VLMEvaluator(BaseEvaluator):
    def __init__(self):
        print("Loading Qwen2-VL-7B-Instruct (SOTA Vision-Language Model)...")
        self.model_id = "Qwen/Qwen2-VL-7B-Instruct"

        # Using bfloat16 saves 50% RAM over float32 with no loss in reasoning quality.
        # Your 64GB RAM will handle this perfectly.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def extract_score(self, text):
        match = re.search(r"Score:\s*(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def evaluate(self, sentences, matched_filenames):
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        SYSTEM_GUIDANCE = (
            "You are a critical evaluator for 'Easy Read' pictograms. Use the full 1-10 scale.\n"
            "Example 1: Text 'Eating' -> Image is just an apple. Score: 5/10 (Object shown, but action is missing).\n"
            "Example 2: Text 'Running' -> Image is a person running. Score: 10/10 (Perfect literal match).\n"
            "Example 3: Text 'Sunny' -> Image is a rain cloud. Score: 1/10 (Incorrect).\n"
            "Aim for a realistic distribution: most good matches are 7-8, only perfect ones are 10."
        )

        for sentence, filenames in zip(sentences, matched_filenames):
            if not filenames:
                continue

            img_path = os.path.join(config.IMAGE_FOLDER, filenames[0])
            image = Image.open(img_path).convert("RGB")

            # Qwen uses a modern Chat Template structure for images and text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{SYSTEM_GUIDANCE}\n\nSentence: '{sentence}'\nProvide your evaluation strictly in this format:\nScore: [X]/10\nReason: [Short explanation]"}
                    ]
                }
            ]

            # Prepare the inputs using the processor's chat template
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cpu")

            # Generate response
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                # Setting do_sample=False is critical for evaluations.
                # It forces the model to pick the most logical tokens instead of "creative" ones.
                do_sample=False
            )

            # Strip the prompt out of the output ids to get just the new text
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
            ]
            clean_response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            results.append({
                "sentence": sentence,
                "image": filenames[0],
                "score": self.extract_score(clean_response),
                "llava_reasoning": clean_response # Kept key name same for your downstream compatibility
            })

            # Print a quick preview to the console
            preview = clean_response.replace('\n', ' ')[:80]
            print(f"Evaluated: {filenames[0]} -> {preview}...")

        out_file = os.path.join(config.OUTPUT_DIR, f"vlm_reasoning_{timestamp}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"VLM Evaluation saved to {out_file}")
