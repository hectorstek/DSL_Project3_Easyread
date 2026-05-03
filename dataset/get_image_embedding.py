import transformers.models.clip.modeling_clip as _clip_mod
if not hasattr(_clip_mod, 'clip_loss') and hasattr(_clip_mod, 'contrastive_loss'):
    _clip_mod.clip_loss = _clip_mod.contrastive_loss

from PIL import Image
from pathlib import Path
import torch
from transformers import AutoModel

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2",
            trust_remote_code=True,
            device_map="auto"  # keep VRAM free for vLLM
        )
        _model.eval()
    return _model


def get_image_embedding(image_path):
    """
    Generates an image embedding using jina-clip-v2 (local, free, no quota).
    Returns a list of floats representing the visual vector.
    """
    import numpy as np
    
    model = _load_model()
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        embedding = model.encode_image(image)
    
    # Handle both numpy arrays and torch tensors
    if isinstance(embedding, np.ndarray):
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    else:
        embedding = embedding.cpu()
        embedding = embedding / embedding.norm()
        return embedding.tolist()