from transformers import CLIPImageProcessor, CLIPModel
import torch
import pandas as pd
from PIL import Image
from io import BytesIO

from search.baseline import BaselineSearcher
import config


class RerankSearcher:
    def __init__(self, baseline: BaselineSearcher):
        self.device = config.DEVICE
        self.baseline = baseline
        self.clip_processor = CLIPImageProcessor.from_pretrained(config.CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL)
        self.clip_model.to(self.device)
        self.index_df = pd.read_csv(config.INDEX_FILE)
        self.data_dir = config.WIKIART_DIR

    def _clip_score(self, query_image, candidate_image) -> float:
        with torch.no_grad():
            q = self.clip_processor(query_image, return_tensors="pt").to(self.device)
            c = self.clip_processor(candidate_image, return_tensors="pt").to(
                self.device
            )
            q_feat = self.clip_model.get_image_features(**q)
            c_feat = self.clip_model.get_image_features(**c)

            q_feat = (
                q_feat
                if isinstance(q_feat, torch.Tensor)
                else (
                    q_feat.image_embeds
                    if hasattr(q_feat, "image_embeds")
                    else q_feat.pooler_output
                )
            )
            c_feat = (
                c_feat
                if isinstance(c_feat, torch.Tensor)
                else (
                    c_feat.image_embeds
                    if hasattr(c_feat, "image_embeds")
                    else c_feat.pooler_output
                )
            )

        # normalize
        q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)
        c_feat = c_feat / c_feat.norm(dim=-1, keepdim=True)

        return (c_feat * q_feat).sum().item()

    def search(
        self, image_bytes: bytes, top_k: int = 10, top_k_reranked: int = 5
    ) -> list:
        candidate_ids = self.baseline.search(image_bytes, top_k)
        query_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        scores = []
        for id in candidate_ids:
            image_path = self.index_df[self.index_df["id"] == id]["filename"].values[0]
            candidate = Image.open(self.data_dir / image_path).convert("RGB")
            score = self._clip_score(query_image, candidate)
            scores.append((id, score))

        reranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k_reranked]
        return [id for id, _ in reranked]

    def close(self):
        pass


if __name__ == "__main__":
    searcher = BaselineSearcher()
    reranker = RerankSearcher(baseline=searcher)
    with open("wikiArt/Baroque/david-teniers-the-younger_kitchen-1644.jpg", "rb") as f:
        image_bytes = f.read()

    top_k_reranked_ids = reranker.search(image_bytes)
    print(top_k_reranked_ids)

    searcher.close()
