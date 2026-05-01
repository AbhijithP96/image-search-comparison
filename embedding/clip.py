from transformers import CLIPImageProcessor, CLIPModel
import torch

MODEL = "openai/clip-vit-base-patch32"
DIMENSION = 512  # CLIP ViT-B/32 has a feature dimension of 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
