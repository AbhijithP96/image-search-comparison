"""
DINOv2 embedding indexing pipeline for the baseline Qdrant collection.

Loads a DINOv2 model, iterates over images listed in a CSV index file, extracts
CLS-token embeddings, and upserts them into a Qdrant collection for subsequent
similarity search. Exposes get_models() and get_embeddings() as reusable helpers
for the search layer.
"""
import argparse
import torch
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

import config

MODEL = config.DINO_MODEL
DIMENSION = config.DINO_DIM
DEVICE = config.DEVICE


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


def get_models():
    # Load the DINO V2 model and processor
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.to(DEVICE)

    return processor, model


def get_embeddings(image, processor, model):
    with torch.no_grad():
        input = processor(image, return_tensors="pt").to(DEVICE)
        output = model(**input)

    # cls token from the last layer of DINO V2
    embeddings = output.last_hidden_state[:, 0, :].squeeze(0).detach().cpu().numpy()
    return embeddings


def main():
    args = get_args()

    # Load the DINO V2 model and processor
    processor, model = get_models()

    # initialize qdrant collection
    client = QdrantClient(path=config.DB_DIR)
    client.create_collection(
        collection_name=config.BASE,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
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

        embeddings = get_embeddings(image, processor, model)

        # add to the vector db
        client.upsert(
            collection_name="baseline",
            points=[
                PointStruct(
                    id=row["id"],
                    vector=embeddings.tolist(),
                    payload={
                        "filename": row["filename"],
                        "genre": row["genre"],
                        "artist": row["artist"],
                    },
                )
            ],
        )

    print(f"Indexed Images: {indexed_count}")
    print(f"Skipped: {skipped}")
    print("Skipped Filenames \n", not_found)

    client.close()


if __name__ == "__main__":
    main()
