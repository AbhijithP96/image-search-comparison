import torch
from qdrant_client import QdrantClient
from PIL import Image
from io import BytesIO
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENT = QdrantClient(path="db")

# Load the DINO V2 model and processor
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)


def get_top_5(image_bytes: bytes) -> list:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    with torch.no_grad():
        input = processor(image, return_tensors="pt").to(DEVICE)
        output = model(**input)

    # cls token from the last layer of DINO V2
    embeddings = output.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()

    results = CLIENT.query_points(
        collection_name="baseline",
        query=embeddings.tolist(),
        limit=5,
        with_payload=True,
    ).points

    return [res.id for res in results]
