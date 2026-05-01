from transformers import CLIPImageProcessor, CLIPModel
import torch

import config

MODEL = config.CLIP_MODEL
DIMENSION = config.CLIP_DIM
DEVICE = config.DEVICE


def get_models():
    # Load the DINO V2 model and processor
    processor = CLIPImageProcessor.from_pretrained(MODEL)
    model = CLIPModel.from_pretrained(MODEL)
    model.to(DEVICE)

    return processor, model


def get_embeddings(image, processor, model):
    with torch.no_grad():
        input = processor(image, return_tensors="pt").to(DEVICE)
        output = model.get_image_features(**input)

    embeddings = (
        output
        if isinstance(output, torch.Tensor)
        else (
            output.image_embeds
            if hasattr(output, "image_embeds")
            else output.pooler_output
        )
    )

    return embeddings.detach().squeeze().cpu().numpy()
