# search/baseline.py
"""
Baseline image similarity search using DINOv2 embeddings and Qdrant.

Provides single and batch query interfaces against a pre-built Qdrant collection,
returning the top-k most similar image IDs ranked by cosine similarity of their
CLS-token embeddings.
"""
import torch
from PIL import Image
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from transformers import AutoImageProcessor, AutoModel

import config
from search.utils import embed_batch


class BaselineSearcher:
    def __init__(self):
        self.device = config.DEVICE
        self.client = QdrantClient(path=str(config.DB_DIR))
        self.collection = config.BASE
        self.processor = AutoImageProcessor.from_pretrained(config.DINO_MODEL)
        self.model = AutoModel.from_pretrained(config.DINO_MODEL)

        self.model.to(self.device)

    def search(self, image_bytes: bytes, top_k: int = 5) -> list:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        with torch.no_grad():
            input = self.processor(image, return_tensors="pt").to(self.device)
            output = self.model(**input)

        # cls token from the last layer of DINO V2
        embeddings = output.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()

        results = self.client.query_points(
            collection_name=self.collection,
            query=embeddings.tolist(),
            limit=top_k,
            with_payload=True,
        ).points

        return [res.id for res in results]

    def search_batch(self, image_bytes_list: list[bytes], top_k: int = 5) -> list[list]:
        embeddings = embed_batch(image_bytes_list, self.processor, self.model)

        requests = [
            QueryRequest(query=emb, limit=top_k, with_payload=True)
            for emb in embeddings
        ]

        batch_results = self.client.query_batch_points(
            collection_name=self.collection, requests=requests
        )

        return [[point.id for point in result.points] for result in batch_results]

    def close(self):
        self.client.close()


if __name__ == "__main__":
    # Example usage
    searcher = BaselineSearcher()
    with open("wikiArt/Baroque/david-teniers-the-younger_kitchen-1644.jpg", "rb") as f:
        image_bytes = f.read()
    top_k_ids = searcher.search(image_bytes)
    print(top_k_ids)
    searcher.close()
