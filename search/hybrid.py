# search/hybrid.py
"""
Hybrid image similarity search combining DINOv2 and CLIP embeddings.

Concatenates L2-normalized DINOv2 and CLIP embeddings
into a single vector, then queries a Qdrant collection built on that joint representation
to retrieve the top-k most similar images.
"""

from embedding import baseline, clip
from PIL import Image
import numpy as np
from qdrant_client import QdrantClient
from io import BytesIO

import config


class HybridSearcher:
    def __init__(self):
        self.dino_processor, self.dino_model = baseline.get_models()
        self.clip_processor, self.clip_model = clip.get_models()

        self.client = QdrantClient(path=config.DB_DIR)
        self.collection = config.HYBRID

    def search(self, image_bytes: bytes, top_k: int = 5) -> list:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        dino_emb = baseline.get_embeddings(image, self.dino_processor, self.dino_model)
        clip_emb = clip.get_embeddings(image, self.clip_processor, self.clip_model)

        # normalize
        dino_emb = dino_emb / np.linalg.norm(dino_emb)
        clip_emb = clip_emb / np.linalg.norm(clip_emb)
        hybrid_emb = np.concatenate((dino_emb, clip_emb))

        # search result
        results = self.client.query_points(
            collection_name=self.collection,
            query=hybrid_emb.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        return [res.id for res in results]

    def close(self):
        self.client.close()


if __name__ == "__main__":
    searcher = HybridSearcher()
    with open("wikiArt/Abstract_Expressionism/mark-tobey_washington.jpg", "rb") as f:
        image_bytes = f.read()
    print(searcher.search(image_bytes))
    searcher.close()
