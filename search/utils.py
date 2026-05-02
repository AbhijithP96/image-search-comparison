from PIL import Image
from io import BytesIO
import torch

import config


def embed_batch(image_byte_list, processor, model) -> list:
    """Create a list of embeddings for batched retrieval"""
    images = [Image.open(BytesIO(b)).convert("RGB") for b in image_byte_list]

    inputs = processor(images, return_tensors="pt").to(config.DEVICE)

    with torch.no_grad():
        output = model(**inputs)

    embeddings = output.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()

    return [embeddings[i].tolist() for i in range(len(images))]
