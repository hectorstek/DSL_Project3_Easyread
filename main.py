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
#from evaluators.top_1_evaluator import GroundTruthEvaluator
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

SENTENCES_TO_GENERATE = [
    "A girl is painting a picture.",
    "I am watering the plants.",
    "The sky is clear and blue.",
    "We are playing a game outside.",
    "I need help with my homework.",
    "A group of friends is laughing together.",
    "The nurse helps patients in the clinic.",
    "The baby is crawling on the floor.",
    "I comb my hair every day.",
    "The child is sitting on a swing.",
    "I wear a coat when it is cold.",
    "He is baking a cake in the oven.",
    "We are going to the park for a picnic.",
    "The firefighter works at the fire station.",
    "I am waiting for my turn.",
    "I am excited about the trip.",
    "Today I feel very hungry.",
    "It is too dark in this hallway.",
    "No running is allowed here.",
    "Please wash your hands before eating.",

    "A man is fixing a bicycle.",
    "The bird is sitting on a branch.",
    "I am drinking a cup of tea.",
    "The dog is barking at the mailman.",
    "We are reading a story together.",
    "She is writing an email.",
    "The baby is giggling softly.",
    "He is closing the curtain.",
    "I am opening the gate.",
    "They are decorating the room.",
    "The bus is stopping at the corner.",
    "I am learning a new language.",
    "The stars are twinkling at night.",
    "The wind is blowing strongly.",
    "We are sitting on a bench.",
    "He is slicing an apple with a knife.",
    "She is sweeping the floor.",
    "I am taking off my hat.",
    "The plane is landing at the airport.",
    "The bell is ringing loudly.",

    "I am texting my sister.",
    "The girl is coloring a drawing.",
    "The boy is throwing a frisbee.",
    "We are dancing to the music.",
    "She is buying a new dress.",
    "He is zipping his backpack.",
    "I am taking a nap.",
    "The butterfly is flying in the garden.",
    "The fish is hiding behind a rock.",
    "We are standing in a circle.",
    "He is repairing the table.",
    "She is ironing a shirt.",
    "I am unpacking my suitcase.",
    "The students are playing outside.",
    "The teacher is grading papers.",
    "I am asking a question.",
    "The chef is cooking dinner.",
    "We are celebrating a birthday.",
    "He is pouring juice.",
    "She is peeling an orange.",

    "I am flipping through a magazine.",
    "The man is riding a motorcycle.",
    "The woman is jogging in the park.",
    "We are visiting a museum.",
    "He is turning a page.",
    "She is blinking her eyes.",
    "I am tying a knot.",
    "The child is clapping happily.",
    "The baby is sucking a pacifier.",
    "We are riding scooters.",
    "He is jumping rope.",
    "She is humming a tune.",
    "I am doodling in my notebook.",
    "The watch is showing the time.",
    "The gate is locked.",
    "We are exiting the theater.",
    "He is entering the classroom.",
    "She is holding a flower.",
    "I am dropping the ball.",
    "The fan is spinning fast.",

    "The fan is slowing down.",
    "We are standing in a line at the store.",
    "He is checking his watch.",
    "She is waving at me.",
    "I am closing the cabinet.",
    "The snacks are on the counter.",
    "We are sharing the cookies.",
    "He is lifting a heavy box.",
    "She is pulling the wagon.",
    "I am putting down the book.",
    "The mirror is shiny and clean.",
    "The grass is wet from the rain.",
    "We are splashing in puddles.",
    "He is holding a raincoat.",
    "She is wiping her face.",
    "I am flipping the calendar.",
    "The movie is starting now.",
    "The concert is ending soon.",
    "We are going to the library.",
    "Good morning and have a nice day."
]

def main():

    #with open(config.GROUND_TRUTH_FILE, "r") as f:
    #    ground_truth_data = json.load(f)
    # comment when not using the recall@k evaluator
    #SENTENCES_TO_GENERATE = [item["generated_sentence"] for item in ground_truth_data]
    #SENTENCES_TO_GENERATE = SENTENCES_TO_GENERATE[:3]

    #matcher = JsonMatcher(
    #    pkl_path="/home/linus/easyread/Linus/Modular/caption_embeddings.pkl",
        #tok_k_filter=20
    #)

    #matcher = FilenameClipMatcher()

    matcher = VLMMatcher() # HybridMatcher() #LLMMatcher() #

    #with open("/home/linus/easyread/Linus/Modular/extractpdf/socialhistories/#EN_COVID19_Protocols_enter_leave_home_manual.json", 'r', encoding='utf-8') as f:
    #    ground_truth_data = json.load(f)

# Extract into the required list format
    #SENTENCES_TO_GENERATE = [item['sentence'] for item in ground_truth_data]

    #matched_files = matcher.match(SENTENCES_TO_GENERATE[0:10], top_k=1)
    output = matcher.match(SENTENCES_TO_GENERATE[0:10], top_k=1)
    matched_files = output[0] #[0]
    scores = output[1] #list(range(2)) #[0, 1] #output[1]
    scores_unsorted = scores
    indices = sorted(range(len(scores)), key=lambda i: scores[i])
    matched_files = [matched_files[i] for i in indices]
    scores = [scores[i] for i in indices]
    #SENTENCES_TO_GENERATE = [SENTENCES_TO_GENERATE[i] + " score: " + str(scores[i]) for i in indices]
    sorted_sentences = [SENTENCES_TO_GENERATE[i] + " score: " + str(scores_unsorted[i]) for i in indices]

    #paired = sorted(zip(output[0], output[1]))
    #matched_files, scores = zip(*paired)
    #matched_files = list(matched_files)
    #scores = list(scores)
    #matched_files = matcher.match(["I want to swim the whole day."], top_k=10)
    print("-------------now matched files")
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
    #date_str = datetime.now().strftime("%Y-%m-%d")
    output_filename = os.path.join(config.OUTPUT_DIR, f"easy_read_{timestamp}.json")
    #output_filename = f"matched_results_{date_str}.json"

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
        #GroundTruthEvaluator(ground_truth_path = "/home/linus/easyread/Linus/Modular/extractpdf/socialhistories/EN_COVID19_Protocols_enter_leave_home_manual.json")
        #ClipEvaluator(),
        #TopKEvaluator(),
        #VLMEvaluator()
    ]

    for evaluator in evaluators:
        evaluator.evaluate(sorted_sentences, matched_files) #[['Kochprogramm.png'], ['Ärztliche Untersuchung.png']]) #[[matched_files[x][0].rsplit('/', 1)[-1]] for x in range(0, 5) ])

if __name__ == "__main__":
    main()
