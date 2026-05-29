from matchers.filename_clip_matcher import FilenameClipMatcher
from matchers.json_matcher import JsonMatcher
from matchers.simple_caption_matcher import SimpleCaptionMatcher
from matchers.hybrid_matcher import HybridMatcher
from matchers.llm_matcher import LLMMatcher
from matchers.vlm_matcher import VLMMatcher
from evaluators.pdf_evaluator import PDFEvaluator
from evaluators.clip_evaluator import ClipEvaluator
from evaluators.top_k_evaluator import TopKEvaluator
from evaluators.vlm_evaluator import VLMEvaluator
import config
import json
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


SENTENCES_TO_GENERATE = [
    #"People have lost their jobs.",
    #"Services that people used, like day centres, have had to close.",
    #"Some people with intellectual disabilities have not been able to get health care.",
    #"People have felt isolated and alone.",
    #"They delivered peer support and kept people with intellectual disabilities connected to each other.",

    "A man is running in the park.",
    "I am eating an apple.",
    "The weather is very sunny today.",
    "We are going for a walk together.",
    "I need help with shopping.",
    "A group of people is talking to each other.",
    "The teacher explains something to the class.",
    "The boy is putting on his shoes.",
    "I wash my hands with soap.",
    "The child is sleeping in a big bed.",
    "I brush my teeth every morning.",
    "She is cooking soup in the kitchen.",
    "We are going to the supermarket to buy food.",
    "The doctor works in a large hospital.",
    "I am waiting for the bus at the station.",
    "I am very happy about the gift.",
    "Today I feel very tired.",
    "It is too loud in this room.",
    "No smoking is allowed here.",
    "Please keep a safe distance.",

    "A woman is reading a book.",
    "The cat is sitting on the chair.",
    "I am drinking a glass of water.",
    "The dog is playing with a ball.",
    "We are watching a movie together.",
    "He is writing a letter.",
    "The baby is crying loudly.",
    "She is opening the window.",
    "I am closing the door.",
    "They are cleaning the room.",
    "The car is stopping at the light.",
    "I am learning a new word.",
    "The sun is rising in the morning.",
    "The moon is shining at night.",
    "We are sitting at the table.",
    "He is cutting bread with a knife.",
    "She is washing the dishes.",
    "I am putting on my jacket.",
    "The train is arriving at the station.",
    "The phone is ringing loudly.",

    "I am calling my friend.",
    "The girl is drawing a picture.",
    "The boy is kicking the ball.",
    "We are listening to music.",
    "She is buying a ticket.",
    "He is opening his bag.",
    "I am taking a shower.",
    "The bird is flying in the sky.",
    "The fish is swimming in the water.",
    "We are standing in a line.",
    "He is fixing the chair.",
    "She is folding clothes.",
    "I am packing my bag.",
    "The children are playing outside.",
    "The teacher is asking a question.",
    "I am answering the question.",
    "The waiter is serving food.",
    "We are eating dinner together.",
    "He is drinking coffee.",
    "She is cutting vegetables.",

    "I am reading the newspaper.",
    "The man is driving a car.",
    "The woman is walking her dog.",
    "We are visiting a friend.",
    "He is opening a book.",
    "She is closing her eyes.",
    "I am tying my shoes.",
    "The child is laughing happily.",
    "The baby is drinking milk.",
    "We are riding bicycles.",
    "He is jumping high.",
    "She is singing a song.",
    "I am writing in my notebook.",
    "The clock is showing the time.",
    "The door is locked.",
    "We are entering the building.",
    "He is leaving the room.",
    "She is holding a cup.",
    "I am dropping the keys.",
    "The light is turning on.",

    "The light is turning off.",
    "We are waiting in the queue.",
    "He is checking his phone.",
    "She is smiling at me.",
    "I am opening the fridge.",
    "The food is on the table.",
    "We are sharing the meal.",
    "He is carrying a box.",
    "She is pushing the cart.",
    "I am picking up the book.",
    "The window is clean.",
    "The floor is wet.",
    "We are walking in the rain.",
    "He is holding an umbrella.",
    "She is drying her hands.",
    "I am turning the page.",
    "The class is starting now.",
    "The lesson is ending soon.",
    "We are going home.",
    "Good night and sleep well."
]

def main():

    #choose matcher, e.g. matcher = FilenameClipMatcher()

    matcher = HybridMatcher() #VLMMatcher()


    output = matcher.match(SENTENCES_TO_GENERATE[0:100], top_k=1)
    matched_files = output[0] #[0]
    scores = output[1]
    scores_unsorted = scores
    indices = sorted(range(len(scores)), key=lambda i: scores[i])
    matched_files = [matched_files[i] for i in indices]
    scores = [scores[i] for i in indices]
    sorted_sentences = [SENTENCES_TO_GENERATE[i] + " score: " + str(scores_unsorted[i]) for i in indices]

    print("The filenames of the matched files:")
    print(matched_files)

    similarities = []

    # Flatten filenames (extract string or empty string)
    filenames_flat = [match[0] if match else "" for match in matched_files]

    # Compute embeddings
    sentence_embeddings = model.encode(sorted_sentences, convert_to_tensor=True)
    filename_embeddings = model.encode(filenames_flat, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(sentence_embeddings, filename_embeddings)

    # Extract diagonal (each sentence with its matched filename)
    for i in range(len(sorted_sentences)):
        sim = cosine_scores[i][i].item()
        similarities.append(sim)

    avg_similarity = float(np.mean(similarities)) if similarities else 0.0
    print(f"Average similarity: {avg_similarity:.4f}")



    results = []
    for sentence, match, sim in zip(sorted_sentences, matched_files, similarities):
        filename = match[0] if match else None

        results.append({
            "sentence": sentence,
            "filename": filename,
            "similarity": sim
        })

    # Create filename with current date
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(config.OUTPUT_DIR, f"easy_read_{timestamp}.json")

    # Save JSON
    output_data = {
        "average_similarity": avg_similarity,
        "results": results
    }
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


    print(f"Results saved to {output_filename}")


    evaluators = [
        PDFEvaluator(),
        #ClipEvaluator(),
        #VLMEvaluator()
    ]

    for evaluator in evaluators:
        evaluator.evaluate(sorted_sentences, matched_files)

if __name__ == "__main__":
    main()
