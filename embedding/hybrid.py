import torch
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from PIL import Image
from tqdm import tqdm
from embedding import baseline, clip
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser("Baseline Model with DINO V2 for Indexing")

    parser.add_argument(
        "--data-dir",
        dest="dir",
        type=str,
        default="wikiArt",
        required=True,
        help="Path to the wikiArt Dataset Folder",
    )
    parser.add_argument(
        "--index-file",
        dest="index",
        type=str,
        default="",
        required=True,
        help="Path to the csv file with image subset.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Load the DINO V2 model and processor
    dino_processor, dino_model = baseline.get_models()
    clip_processor, clip_model = clip.get_models()

    size = baseline.DIMENSION + clip.DIMENSION

    # initialize qdrant collection
    client = QdrantClient(path="hybrid_db")
    client.create_collection(
        collection_name="hybrid",
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
    )

    # Load the index file
    index_df = pd.read_csv(args.index, encoding="utf-8")

    # total indexed and skipped
    indexed_count, skipped = 0, 0
    not_found = []

    for idx, row in tqdm(index_df.iterrows(), desc="Indexing", total=len(index_df)):
        image_path = Path(args.dir) / row["filename"]

        try:
            image = Image.open(image_path).convert("RGB")
            indexed_count += 1
        except FileNotFoundError:
            skipped += 1
            not_found.append(row["filename"])
            continue

        dino_embedding = baseline.get_embeddings(image, dino_processor, dino_model)
        clip_embedding = clip.get_embeddings(image, clip_processor, clip_model)

        # normalize the embeddings
        dino_embedding = dino_embedding / np.linalg.norm(dino_embedding)
        clip_embedding = clip_embedding / np.linalg.norm(clip_embedding)

        # concatenate the embeddings
        hybrid_embedding = np.concatenate((dino_embedding, clip_embedding))

        # create a point struct and upsert to qdrant
        payload = {
            "filename": row["filename"],
            "genre": row["genre"],
            "artist": row["artist"],
        }
        point = PointStruct(
            id=row["id"], vector=hybrid_embedding.tolist(), payload=payload
        )
        client.upsert(collection_name="hybrid", points=[point])
    print(f"Indexed Images: {indexed_count}")
    print(f"Skipped: {skipped}")
    print("Skipped Filenames \n", not_found)

    client.close()


if __name__ == "__main__":
    main()
