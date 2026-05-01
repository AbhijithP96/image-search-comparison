from embedding import baseline, clip
from PIL import Image
import numpy as np
from qdrant_client import QdrantClient
from io import BytesIO

dino_processor, dino_model = baseline.get_models()
clip_processor, clip_model = clip.get_models()

CLIENT = QdrantClient(path="hybrid_db")


def get_top_5(image_bytes: bytes) -> list:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    dino_embeddings = baseline.get_embeddings(image, dino_processor, dino_model)
    clip_embeddings = clip.get_embeddings(image, clip_processor, clip_model)

    # normalize the embeddings
    dino_embeddings = dino_embeddings / np.linalg.norm(dino_embeddings)
    clip_embeddings = clip_embeddings / np.linalg.norm(clip_embeddings)

    # concatenate the embeddings
    hybrid_embeddings = np.concatenate((dino_embeddings, clip_embeddings))

    results = CLIENT.query_points(
        collection_name="hybrid",
        query=hybrid_embeddings.tolist(),
        limit=5,
        with_payload=True,
    ).points

    return [res.id for res in results]


if __name__ == "__main__":
    # Example usage
    with open("wikiArt/Abstract_Expressionism/mark-tobey_washington.jpg", "rb") as f:
        image_bytes = f.read()
    top_5_ids = get_top_5(image_bytes)
    print(top_5_ids)

    CLIENT.close()
