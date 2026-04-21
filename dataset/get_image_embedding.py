from PIL import Image
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel

# Load once at module level (cached after first call)
_model = None
_processor = None

def _load_model():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()
    return _model, _processor


def get_image_embedding(image_path):
    """
    Generates an image embedding using CLIP (local, free, no quota).
    Returns a list of floats representing the visual vector.
    """
    model, processor = _load_model()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize

    return embedding[0].tolist()