from scipy.spatial.distance import cosine
import pickle
import mimetypes
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

# path to vectors
EMBEDDINGS_PATH = Path("gold_embeddings.pkl")
DATA_DIR = Path("easyread-retrieval-dataset/data")
METADATA_PATH = Path("easyread-retrieval-dataset/metadata_v2.jsonl")
GOLD_PIC_DIR = Path("easyread-retrieval-dataset/gold_standard_pic")

with open(EMBEDDINGS_PATH, "rb") as f:
    gold_library_vectors = pickle.load(f)

# map absolute paths to cluster paths
for key, data in gold_library_vectors.items():
    old_path = Path(data["path"])
    data["path"] = str(GOLD_PIC_DIR / old_path.name)


def load_already_labeled() -> set[str]:
    """Return the set of file_name values already present in metadata.jsonl."""
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
    

def find_best_examples(target_image_path, top_k=2):
    """Find the two best examples from the Gold Standard Dataset, for LLM input."""
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


# For API run purposes instead of locally
'''
_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _read_env_value(name: str):
    v = os.getenv(name)
    if v:
        return v.strip().strip('"').strip("'")
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return None


def _get_groq_api_key():
    key = _read_env_value("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Get a free key at https://console.groq.com/ "
            "and set GROQ_API_KEY in your environment or Hector/.env."
        )
    return key

def _get_groq_api_key():
    return _read_env_value("GROQ_API_KEY") or "local"


GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
# Default urllib User-Agent is often blocked by Cloudflare (HTTP 403, error code 1010).
_GROQ_USER_AGENT = os.getenv(
    "GROQ_HTTP_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)
GROQ_MAX_RETRIES = max(1, int(os.getenv("GROQ_MAX_RETRIES", "12")))
'''


VLLM_MAX_RETRIES = max(1, int(os.getenv("VLLM_MAX_RETRIES", "3")))
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "RedHatAI/Qwen2.5-VL-7B-Instruct-quantized.w4a16")
GROQ_CHAT_URL = os.getenv("VLLM_CHAT_URL", "http://localhost:8000/v1/chat/completions")


def _pil_to_data_url(image: Image.Image) -> str:
    """Turns a PIL image object into a base64 data URL string that can be embed"""
    fmt = (image.format or "PNG").upper()
    if fmt not in ("PNG", "JPEG", "JPG", "WEBP", "GIF"):
        fmt = "PNG"
    save_fmt = "JPEG" if fmt in ("JPEG", "JPG") else fmt
    buf = BytesIO()
    image.save(buf, format=save_fmt)
    mime = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }[save_fmt]
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _path_to_data_url(path: str) -> str:
    """Return the image file from the corresponding path."""
    with open(path, "rb") as f:
        raw = f.read()
    mt = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mt};base64,{b64}"


def _groq_json_from_images(
    user_content: list,
    *,
    api_key: str,
    model: str = GROQ_VISION_MODEL,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.2,
            "max_tokens": 1024,  # vLLM uses max_tokens, not max_completion_tokens
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = Request(
        GROQ_CHAT_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    for attempt in range(VLLM_MAX_RETRIES):
        try:
            with urlopen(req, timeout=120) as resp:
                body = json.load(resp)
            break
        except HTTPError as e:
            err_txt = e.read().decode("utf-8", errors="replace")
            if attempt < VLLM_MAX_RETRIES - 1:
                print(f"  HTTP {e.code}, retrying (attempt {attempt + 1}/{VLLM_MAX_RETRIES})")
                time.sleep(2.0)
                continue
            raise RuntimeError(f"vLLM API HTTP {e.code}: {err_txt}") from e
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as ex:
        raise RuntimeError(f"Unexpected vLLM response: {body!r}") from ex


def get_label_via_rag(
    target_image_path,
    example_images,
    example_jsons,
    *,
    target_image_name: str | None = None,
    example_image_names: list[str] | None = None,
):
    task_text = """Task: Label this Easy Read pictogram into JSON with this exact structure:
{"raw_caption": "<one sentence describing the image>", "intent": {"actors": [...], "actions": [...], "objects": [...], "setting": "<context or 'unknown'>"}}
Output only a single JSON object, no markdown. Focus on the meaning of the pictogram. The filename is provided as auxiliary context; use it only to disambiguate when consistent with the visual content, and never invent details not visible in the image.
"""

    ex0 = example_jsons[0]
    ex1 = example_jsons[1]
    if not isinstance(ex0, str):
        ex0 = json.dumps(ex0, ensure_ascii=False)
    if not isinstance(ex1, str):
        ex1 = json.dumps(ex1, ensure_ascii=False)
    target_name = target_image_name or Path(target_image_path).name
    ex_names = example_image_names or [
        f"example_{i + 1}.jpg" for i in range(len(example_images))
    ]

    user_content = [
        {"type": "text", "text": task_text},
        {"type": "text", "text": f"Example 1 (filename: {ex_names[0]}):"},
        {
            "type": "image_url",
            "image_url": {"url": _pil_to_data_url(example_images[0])},
        },
        {"type": "text", "text": f"Result: {ex0}"},
        {"type": "text", "text": f"Example 2 (filename: {ex_names[1]}):"},
        {
            "type": "image_url",
            "image_url": {"url": _pil_to_data_url(example_images[1])},
        },
        {"type": "text", "text": f"Result: {ex1}"},
        {"type": "text", "text": f"Now label this image (filename: {target_name}):"},
        {
            "type": "image_url",
            "image_url": {"url": _path_to_data_url(target_image_path)},
        },
    ]
    return _groq_json_from_images(user_content, api_key=_get_groq_api_key())


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_IMAGES = 12000

already_labeled = load_already_labeled()
image_paths = sorted([
    p for p in DATA_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
])
image_paths_to_process = image_paths[:MAX_IMAGES]

print(
    f"Found {len(image_paths)} images. "
    f"Processing first {len(image_paths_to_process)}. "
    f"{len(already_labeled)} already labeled."
)
print(f"Label LLM: vLLM ({GROQ_VISION_MODEL})")

with open(METADATA_PATH, "a", encoding="utf-8") as out_f:
    for i, image_path in enumerate(image_paths_to_process):
        file_name = str(DATA_DIR / image_path.name)

        if file_name in already_labeled:
            print(f"[{i+1}/{len(image_paths_to_process)}] Skipping (already labeled): {image_path.name}")
            continue

        print(f"[{i+1}/{len(image_paths_to_process)}] Labeling: {image_path.name}")

        try:
            best_matches = find_best_examples(str(image_path), top_k=2)

            if len(best_matches) < 2:
                print(f"  WARNING: Not enough retrieved examples, skipping.")
                continue

            example_images = [Image.open(m["path"]) for m in best_matches]
            example_jsons = [m["intent"] for m in best_matches]

            result_json_str = get_label_via_rag(
                str(image_path),
                example_images,
                example_jsons,
                target_image_name=image_path.name,
                example_image_names=[Path(m["path"]).name for m in best_matches],
            )
            result = json.loads(result_json_str)

            entry = {
                "file_name": file_name,
                "raw_caption": result.get("raw_caption", ""),
                "intent": result.get("intent", {})
            }

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()  # ensure it's written even if the script crashes mid-run

        except Exception as e:
            print(f"  ERROR on {image_path.name}: {e}")
            continue

print("Done.")