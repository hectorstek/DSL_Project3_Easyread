import json
import os
from datetime import datetime
from evaluators.base_evaluator import BaseEvaluator
import config
import json

class TopKEvaluator(BaseEvaluator):
    def __init__(self):
        with open(config.GROUND_TRUTH_FILE, 'r') as f:
            self.ground_truth = json.load(f)

    def evaluate(self, sentences: list[str], matched_filenames: list[list[str]]) -> None:
        hits = 0
        total = len(sentences)
        matches = []

        for i, sentence in enumerate(sentences):
            # list of ground truth images for that sentence
            match = [item["true_image"] for item in self.ground_truth if item["generated_sentence"].strip() == sentence.strip()]
            if not match:
                print(f"Warning: No ground truth found for sentence: {sentence[:30]}...")
                continue
            true_image = match[0]

            # Check if it exists anywhere in the top-k list
            if true_image in matched_filenames[i]:
                hits += 1
                matches.append(matched_filenames[i].index(true_image))
                matches.append(sentence.strip())

        accuracy = (hits / total) * 100
        print(f"Top-K Recall: {accuracy:.2f}% ({hits}/{total} matches found)")

        # Save results with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(config.OUTPUT_DIR, f"recall_results_{ts}.json"), "w") as f:
            json.dump({"recall_score": accuracy, "hits": hits, "total": total, "matches": matches}, f)
