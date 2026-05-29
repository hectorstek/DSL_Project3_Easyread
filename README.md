# EasyRead Image Retrieval

**ETH Zurich Data Science Lab**
**Structured Intent and Image Retrieval for Easy Read Content**

Retrieve pictograms from the ARASAAC library to illustrate EasyRead sentences. The pipeline extracts structured intent (actors, actions, objects, setting, emotion) from an input sentence, scores all pictograms in the corpus by weighted multi-field similarity, and a vision-language model reranks the top candidates to return the final image.

This project was carried out in collaboration with UNICEF, in the context of accessibility research for users with intellectual disabilities, low literacy, or language barriers.

---

## Why this matters

EasyRead is an accessibility framework where each sentence is paired with a clear, simple pictogram to support comprehension. Producing these materials manually is slow, expensive, and requires expert curation. Conventional image-text retrieval (e.g., CLIP) operates on holistic similarity and tends to flatten the compositional structure of a sentence, missing the distinctions between _who_ is acting, _what_ is being done, and the _context_. Our approach addresses this by retrieving over structured intent fields rather than dense sentence embeddings, while restricting the search to ARASAAC pictograms that already satisfy EasyRead design principles by construction.

---

## Pipeline overview

The system has two stages:

### Stage 1 — Offline labelling (run once)

Every ARASAAC image is annotated with a structured JSON label of the form:

```json
{
  "raw_caption": "A short sentence describing the image.",
  "intent": {
    "actors": ["..."],
    "actions": ["..."],
    "objects": ["..."],
    "setting": "...",
    "emotion": "positive | neutral | negative"
  }
}
```

This is done with **Qwen3-VL-8B-Instruct-FP8** (served via vLLM), guided by retrieval-augmented prompting from a 50-image gold standard set curated by the authors. A self-verification step then reduces hallucinations by asking the model to re-check its own annotation against the image.

The full corpus is then encoded with **paraphrase-multilingual-MiniLM-L12-v2** (SentenceTransformers) and stored as a `.pkl` index for fast lookup at runtime.

### Stage 2 — Runtime retrieval

For each input sentence:

1. **Intent extraction.** `Qwen2.5-1.5B-Instruct` parses the sentence into the same JSON schema used for labelling.
2. **Encoding.** Each field (caption + 5 intent components) is encoded with the same MiniLM model.
3. **Per-field similarity.** Cosine similarity is computed between the sentence and every entry in the indexed corpus, field by field.
4. **Weighted ranking.** Field similarities are aggregated using empirically chosen weights (50% caption, 20% action, 15% actor, 10% object, 5% emotion) to produce the top-5 candidates.
5. **VLM reranking.** **Gemma 4 E4B-it** sees the top-5 alongside the sentence, picks the best match, and returns a confidence score.

The final output is a single ARASAAC pictogram plus a confidence score that flags low-quality matches for human review.

---

## Repository structure

```
DSL_Project3_Easyread/
├── dataset/               # offline corpus annotation (RAG + self-verification)
├── matchers/              # runtime retrieval pipeline
├── app/                   # Streamlit interface
├── evaluators/            # evaluation scripts
├── main.py                # main runner with 100 input sentences
├── requirements.txt
└── README.md
```

---

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/hectorstek/DSL_Project3_Easyread.git
cd DSL_Project3_Easyread
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### ARASAAC dataset

The ARASAAC pictograms are not included in this repository. Download them from [arasaac.org](https://arasaac.org) (CC BY-NC-SA, © Government of Aragon) and place them under `data/arasaac/`.

### Models

All models are downloaded automatically from Hugging Face on first run. For the labelling stage you will need a GPU (we used an NVIDIA RTX 5060 Ti, 16 GB). The retrieval stage runs on CPU, but very slowly.

---

## Usage

### Run the Streamlit app

The easiest way to try the pipeline interactively:

```bash
streamlit run app/app.py
```

Type an EasyRead sentence into the interface and see the retrieved pictogram with its confidence score.

### Run retrieval on a batch of sentences

```bash
python retrieval/run.py --input sentences.txt --output results/
```

`sentences.txt` should contain one sentence per line.

### Re-run labelling on the corpus

If you want to regenerate annotations (for a different model, schema, or dataset):

```bash
python labelling/run.py --images data/arasaac/ --output data/labels.jsonl
```

This requires a vLLM server running locally — see `labelling/README.md` for details.

---

## Evaluation

We evaluated the pipeline using three complementary methods:

- **CLIP similarity** between input sentence and retrieved image
- **VLM-as-a-judge** scoring using Gemma 4 E4B-it
- **Human user study** with 4 participants rating 100 sentence-image pairs on a 0–5 scale

Full results and qualitative comparisons against existing prototypes (UNICEF, GlobalSymbols EasyMaker) are reported in the paper.

---

## Authors

- **Linus Reul** — Department of Computer Science, ETH Zurich
- **Hector Stekelorom** — Department of Computer Science, ETH Zurich

### Project supervisors

- Sonia Laguna (ETH Zurich)
- Emanuele Palumbo (ETH Zurich)
- Thy Nowak-Tran (UNICEF Digital Impact Division)
- Julia Vogt (ETH Zurich)

### Course organisers

- Paola Malsot
- Arnout Devos

---

## License

This code is released under the **MIT License**. See `LICENSE` for details.

ARASAAC pictograms are © Government of Aragon and distributed under the Creative Commons license **CC BY-NC-SA** by Sergio Palao via [arasaac.org](https://arasaac.org). They are not redistributed here.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{reul2026easyread,
  title  = {Structured Intent and Image Retrieval for Easy Read Content},
  author = {Hector Stekelorom and Linus Reul},
  year   = {2026},
  note   = {263-3300-10L Data Science Lab, ETH Zurich},
  url    = {https://github.com/hectorstek/DSL_Project3_Easyread}
}
```
