from matchers.filename_clip_matcher import FilenameClipMatcher
from matchers.json_matcher import JsonMatcher
from matchers.simple_caption_matcher import SimpleCaptionMatcher
from matchers.hybrid_matcher import HybridMatcher
from evaluators.pdf_evaluator import PDFEvaluator
from evaluators.clip_evaluator import ClipEvaluator
from evaluators.top_k_evaluator import TopKEvaluator
from evaluators.vlm_evaluator import VLMEvaluator
#from evaluators.top_1_evaluator import GroundTruthEvaluator
import config
import json


SENTENCES_TO_GENERATE = [
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

    matcher = HybridMatcher()

    #with open("/home/linus/easyread/Linus/Modular/extractpdf/socialhistories/#EN_COVID19_Protocols_enter_leave_home_manual.json", 'r', encoding='utf-8') as f:
    #    ground_truth_data = json.load(f)

# Extract into the required list format
    #SENTENCES_TO_GENERATE = [item['sentence'] for item in ground_truth_data]

    matched_files = matcher.match(SENTENCES_TO_GENERATE, top_k=1)
    #matched_files = matcher.match(["I want to swim the whole day."], top_k=10)
    print("-------------now matched files")
    print(matched_files)


    evaluators = [
        PDFEvaluator(),
        #GroundTruthEvaluator(ground_truth_path = "/home/linus/easyread/Linus/Modular/extractpdf/socialhistories/EN_COVID19_Protocols_enter_leave_home_manual.json")
        ClipEvaluator(),
        #TopKEvaluator(),
        VLMEvaluator()
    ]

    for evaluator in evaluators:
        evaluator.evaluate(SENTENCES_TO_GENERATE, matched_files) #[['Kochprogramm.png'], ['Ärztliche Untersuchung.png']]) #[[matched_files[x][0].rsplit('/', 1)[-1]] for x in range(0, 5) ])

if __name__ == "__main__":
    main()
