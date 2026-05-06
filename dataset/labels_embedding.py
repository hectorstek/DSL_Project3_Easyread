from scipy.spatial.distance import cosine
import pickle
import os
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from PIL import Image
from get_image_embedding import get_image_embedding

# ── Paths ──────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = Path("gold_embeddings.pkl")
DATA_DIR        = Path("easyread-retrieval-dataset/data")
METADATA_PATH   = Path("easyread-retrieval-dataset/metadata_v5.jsonl")
GOLD_PIC_DIR    = Path("easyread-retrieval-dataset/gold_standard_pic")

# ── Config ─────────────────────────────────────────────────────────────────────
VLLM_MAX_RETRIES  = max(1, int(os.getenv("VLLM_MAX_RETRIES", "3")))
VLLM_MODEL        = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct-FP8")
VLLM_CHAT_URL     = os.getenv("VLLM_CHAT_URL", "http://localhost:8000/v1/chat/completions")
MAX_IMAGES        = 16000
TOP_K             = 2
THUMBNAIL_SIZE    = (256, 256)

# ── Load gold embeddings ───────────────────────────────────────────────────────
with open(EMBEDDINGS_PATH, "rb") as f:
    gold_library_vectors = pickle.load(f)

if not gold_library_vectors:
    raise RuntimeError(
        f"Gold embeddings pickle is empty: {EMBEDDINGS_PATH}\n"
        f"Run build_gold_embeddings.py first."
    )

for key, data in gold_library_vectors.items():
    old_path = Path(data["path"])
    data["path"] = str(GOLD_PIC_DIR / old_path.name)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_already_labeled() -> set[str]:
    labeled = set()
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    labeled.add(entry["file_name"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return labeled


def find_best_examples(target_image_path, top_k=TOP_K):
    target_vec = get_image_embedding(target_image_path)
    scores = []
    for _, data in gold_library_vectors.items():
        similarity = 1 - cosine(target_vec, data["vector"])
        scores.append({
            "path": data["path"],
            "intent": data["label"],
            "score": similarity
        })
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


def _pil_to_data_url(image: Image.Image) -> str:
    img = image.copy()
    img.thumbnail(THUMBNAIL_SIZE)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _path_to_data_url(path: str) -> str:
    img = Image.open(path)
    img.thumbnail(THUMBNAIL_SIZE)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _vllm_call(user_content: list) -> str:
    """Sync call to vLLM, returns the message content string."""
    payload = json.dumps({
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},  # disable thinking mode
    }).encode("utf-8")

    req = Request(
        VLLM_CHAT_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer local",
        },
        method="POST",
    )

    for attempt in range(VLLM_MAX_RETRIES):
        try:
            with urlopen(req, timeout=180) as resp:
                body = json.load(resp)
            return body["choices"][0]["message"]["content"]
        except HTTPError as e:
            err_txt = e.read().decode("utf-8", errors="replace")
            if attempt < VLLM_MAX_RETRIES - 1:
                print(f"  HTTP {e.code}, retrying ({attempt + 1}/{VLLM_MAX_RETRIES})")
                time.sleep(2.0)
                continue
            raise RuntimeError(f"vLLM HTTP {e.code}: {err_txt}") from e
    raise RuntimeError("vLLM: exhausted retries")


# ── Step 1: RAG labelling ─────────────────────────────────────────────────────

def label_with_rag(target_image_path, example_images, example_jsons, target_image_name, example_image_names) -> dict:
    task_text = """Task: Label this Easy Read pictogram into JSON with this exact structure:
    {"raw_caption": "<one sentence describing the image>", "intent": {"actors": [...], "actions": [...], "objects": [...], "setting": "<context or 'unknown'>", "emotion": "<one of: positive, negative, neutral>"}}

    Output only a single JSON object, no markdown.

    Important guidance on filenames:
    - The filename is often a strong semantic hint about what the pictogram represents.
    - If the filename is descriptive (e.g. "baby_suckling", "fire_extinguisher", "happy_birthday"), USE IT to disambiguate ambiguous visuals.
    - Only ignore the filename when it is clearly generic (e.g. "image_2", "icon_42") or contradicts what is unmistakably visible.
    - When the filename and image agree, lean into the filename's interpretation.
    - When in doubt, the filename usually wins over weak visual cues.

    Emotion reflects the affective tone: positive (happiness/celebration), negative (distress/pain), or neutral (everything else)."""

    serialized = [
        ej if isinstance(ej, str) else json.dumps(ej, ensure_ascii=False)
        for ej in example_jsons
    ]

    content = [{"type": "text", "text": task_text}]
    for idx, (img, ej, name) in enumerate(zip(example_images, serialized, example_image_names)):
        content += [
            {"type": "text", "text": f"Example {idx + 1} (filename: {name}):"},
            {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}},
            {"type": "text", "text": f"Result: {ej}"},
        ]
    content += [
        {"type": "text", "text": f"Now label this image (filename: {target_image_name}):"},
        {"type": "image_url", "image_url": {"url": _path_to_data_url(target_image_path)}},
    ]

    raw = _vllm_call(content)
    return json.loads(raw)


# ── Step 2: Verification ──────────────────────────────────────────────────────

def verify_label(target_image_path, label, target_image_name) -> dict:
    verify_text = f"""You previously labeled this Easy Read pictogram (filename: {target_image_name}) as:
{json.dumps(label, ensure_ascii=False)}

Look at the image again carefully. Does this label accurately describe ONLY what you can SEE in the image?
- If yes, return the exact same JSON unchanged.
- If no, return a corrected JSON with this exact structure:
{{"raw_caption": "<one sentence>", "intent": {{"actors": [...], "actions": [...], "objects": [...], "setting": "<context or unknown>", "emotion": "<positive|negative|neutral>"}}}}
Output only a single JSON object, no markdown."""

    content = [
        {"type": "text", "text": verify_text},
        {"type": "image_url", "image_url": {"url": _path_to_data_url(target_image_path)}},
    ]
    try:
        raw = _vllm_call(content)
        return json.loads(raw)
    except Exception as e:
        print(f"  WARNING: verification failed ({e}), keeping original label")
        return label


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    already_labeled = load_already_labeled()
    image_paths = sorted([
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        and not p.name.startswith("._")
    ])
    image_paths_to_process = [
        p for p in image_paths[:MAX_IMAGES]
        if str(DATA_DIR / p.name) not in already_labeled
    ]

    print(f"Found {len(image_paths)} images.")
    print(f"Already labeled: {len(already_labeled)}.")
    print(f"To process: {len(image_paths_to_process)}.")
    print(f"Label LLM: vLLM ({VLLM_MODEL})")
    print(f"Settings: top_k={TOP_K}, thumbnail={THUMBNAIL_SIZE}")

    with open(METADATA_PATH, "a", encoding="utf-8") as out_f:
        for i, image_path in enumerate(image_paths_to_process):
            file_name = str(DATA_DIR / image_path.name)
            print(f"[{i+1}/{len(image_paths_to_process)}] {image_path.name}")

            try:
                # 1. RAG examples
                best = find_best_examples(str(image_path), top_k=TOP_K)
                example_images = [Image.open(m["path"]) for m in best]
                example_jsons  = [m["intent"] for m in best]
                example_names  = [Path(m["path"]).name for m in best]

                # 2. RAG labelling
                label = label_with_rag(
                    str(image_path),
                    example_images,
                    example_jsons,
                    image_path.name,
                    example_names,
                )

                # 3. Verification
                label = verify_label(str(image_path), label, image_path.name)

                entry = {
                    "file_name": file_name,
                    "raw_caption": label.get("raw_caption", ""),
                    "intent": label.get("intent", {}),
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()

            except Exception as e:
                print(f"  ERROR on {image_path.name}: {e}")
                continue

    print("Done.")


if __name__ == "__main__":
    main()