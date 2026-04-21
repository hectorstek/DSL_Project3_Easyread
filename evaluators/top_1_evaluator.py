import json
import os
from abc import ABC, abstractmethod
from evaluators.base_evaluator import BaseEvaluator

class GroundTruthEvaluator(BaseEvaluator):
    def __init__(self, ground_truth_path: str, output_path: str = "evaluation_results_1504.json"):
        """
        Args:
            ground_truth_path: Path to the JSON containing sentences and their correct filenames.
            output_path: Where the final evaluation JSON will be saved.
        """
        self.output_path = output_path
        
        # Load the ground truth data and create a quick lookup dictionary
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.ground_truth_map = {item['sentence']: item['filenames'] for item in raw_data}

    def evaluate(self, sentences: List[str], matched_filenames: list[list[str]]) -> None:
        total_sentences = len(sentences)
        total_matches = 0
        details = []

        for sentence, retrieved_list in zip(sentences, matched_filenames):
            ground_truth_files = self.ground_truth_map.get(sentence, [])
            
            # Assuming we evaluate based on the top-1 retrieved filename
            retrieved_filename = retrieved_list[0] if retrieved_list else None
            
            # Check if the retrieved filename is in the list of ground truth filenames
            is_match = retrieved_filename in ground_truth_files
            
            if is_match:
                total_matches += 1
                
            details.append({
                "sentence": sentence,
                "retrieved_filename": retrieved_filename,
                "is_match": is_match,
                "ground_truth_filenames": ground_truth_files
            })

        # Compile final results
        results = {
            "total_sentences": total_sentences,
            "total_matches": total_matches,
            "accuracy": round((total_matches / total_sentences) * 100, 2) if total_sentences > 0 else 0,
            "details": details
        }

        # Save to JSON
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"Evaluation complete: {total_matches}/{total_sentences} matches.")
        print(f"Results saved to {self.output_path}")
