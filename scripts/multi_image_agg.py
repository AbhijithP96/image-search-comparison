import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from embedding.baseline import get_models, get_embeddings

DATA_DIR = Path("met")
LABEL = DATA_DIR / "met.csv"
Path("agg_db").mkdir(exist_ok=True, parents=True)
CLIENT = QdrantClient(path="agg_db")

processor, model = get_models()


def get_index_images():
    df = pd.read_csv(LABEL)
    df = df[df["subset"] == "index"]

    return df


def get_query_images():
    df = pd.read_csv(LABEL)
    return df[df["subset"] == "query"]


def get_query_embedding(filename: str) -> np.ndarray:
    image = Image.open(Path(filename)).convert("RGB")
    embedding = get_embeddings(image, processor, model)
    return embedding


def generate_avg_pooled_embed(df: pd.DataFrame, client: QdrantClient):

    for art_id, (title, group) in enumerate(df.groupby("title")):
        embeddings = []

        payload = {
            "title": title,
            "department": group.iloc[0]["department"],
            "class": group.iloc[0]["class"],
            "filename": group.iloc[0]["filename"],
        }

        for _, row in group.iterrows():

            filepath = Path(row["filename"])

            image = Image.open(filepath).convert("RGB")
            embed = get_embeddings(image, processor, model)

            point = PointStruct(id=row["id"], vector=embed.tolist(), payload=payload)

            client.upsert(collection_name="single", points=[point])

            embed = embed / np.linalg.norm(embed)
            embeddings.append(embed)

        avg_pooled_embed = np.mean(embeddings, axis=0)

        point = PointStruct(
            id=art_id, vector=avg_pooled_embed.tolist(), payload=payload
        )

        client.upsert(collection_name="pooled", points=[point])


def top1_accuracy(retrieved, query_label, label_field="department"):
    if not retrieved:
        return 0
    return 1 if retrieved[0].payload[label_field] == query_label else 0


def precision_at_k(retrieved, query_label, k=5, label_field="department"):
    relevant = sum(1 for r in retrieved[:k] if r.payload[label_field] == query_label)
    return relevant / k


def evaluate(label_field="class", top_k=2, client=None):
    query_df = get_query_images()

    s_top1s = []
    s_p5s = []
    p_top1s = []
    p_p5s = []

    for _, row in query_df.iterrows():
        query_label = row[label_field]
        query_filename = row["filename"]
        query_title = row["title"]

        # Get embedding
        query_embedding = get_query_embedding(query_filename)

        # Search both collections
        single_results = client.query_points(
            collection_name="single",
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        pooled_results = client.query_points(
            collection_name="pooled",
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        ).points

        # Compute metrics
        s_t1 = top1_accuracy(single_results, query_label, label_field)
        s_p5 = precision_at_k(single_results, query_label, top_k, label_field)

        s_top1s.append(s_t1)
        s_p5s.append(s_p5)

        p_t1 = top1_accuracy(pooled_results, query_label, label_field)
        p_p5 = precision_at_k(pooled_results, query_label, top_k, label_field)

        p_top1s.append(p_t1)
        p_p5s.append(p_p5)

    # Summary
    mean_top1_single = np.mean(s_top1s)
    mean_p5_single = np.mean(s_p5s)

    mean_top1_pooled = np.mean(p_top1s)
    mean_p5_pooled = np.mean(p_p5s)

    print(f"Results for top_k = {top_k}")
    print("=" * 55)
    print(f"Label field    : {label_field}")
    print(f"Queries        : {len(query_df)}")
    print(f"                  Single \t Pooled")
    print(f"Top-1 Accuracy : {mean_top1_single:.4f} \t {mean_top1_pooled:.4f}")
    print(f"Mean P@5       : {mean_p5_single:.4f} \t {mean_p5_pooled:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "create":
        df = get_index_images()

        CLIENT.create_collection(
            collection_name="pooled",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        CLIENT.create_collection(
            collection_name="single",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        generate_avg_pooled_embed(df, CLIENT)
    elif sys.argv[1] == "eval":
        evaluate(client=CLIENT, top_k=2)
        evaluate(client=CLIENT, top_k=3)
        evaluate(client=CLIENT, top_k=5)
    else:
        print("Not a valid argument. Provide Either create or eval.")

    CLIENT.close()
