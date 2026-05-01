import time
import pandas as pd
import requests
from pathlib import Path

QUERY = "dataset/csv/query_set.csv"
INDEX = "dataset/csv/index_set.csv"

DATA_DIR = Path("wikiArt")
URL = "http://localhost:8000/search"


def get_retrieved_styles(retrieved_ids, index_df):
    retrieved_style = []

    for rid in retrieved_ids:
        row = index_df[index_df["id"] == rid]

        if not row.empty:
            style = row["genre"].iloc[0].strip()
            retrieved_style.append(style)

    return retrieved_style


def precison_at_k(retrieved_style, query_style, k=5):

    relevant = sum(1 for style in retrieved_style[:k] if style == query_style)

    return relevant / k


def average_precision_at_k(retrieved_style, query_style, k=5):
    relevant_count = 0
    precison = []

    for i, style in enumerate(retrieved_style[:k]):
        if style == query_style:
            relevant_count += 1
            precison.append(relevant_count / (i + 1))

    if not precison:
        return 0.0

    return sum(precison) / len(precison)


def accuracy_at_1(retrieved_style, query_style):
    return 1 if retrieved_style[0] == query_style else 0


def eval():

    query_df = pd.read_csv(QUERY)
    index_df = pd.read_csv(INDEX)

    p5 = []
    ap5 = []
    top1 = []
    latency = []

    for idx, row in query_df.iterrows():
        image_path = DATA_DIR / row["filename"]

        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            start_time = time.perf_counter()
            response = requests.post(URL, files=files)
            end_time = time.perf_counter()

        retrieved_style = get_retrieved_styles(response.json()["ids"], index_df)

        precison_at_k_score = precison_at_k(retrieved_style, row["genre"].strip())
        avg_precision_at_k_score = average_precision_at_k(
            retrieved_style, row["genre"].strip()
        )
        acc_at_1_score = accuracy_at_1(retrieved_style, row["genre"].strip())

        p5.append((row["id"], precison_at_k_score))
        ap5.append((row["id"], avg_precision_at_k_score))
        top1.append((row["id"], acc_at_1_score))
        latency.append((row["id"], end_time - start_time))

    return {
        "precision_at_5": p5,
        "average_precision_at_5": ap5,
        "accuracy_at_1": top1,
        "latency": latency,
    }


if __name__ == "__main__":
    metrics = eval()

    print("Precision@5:")
    for id, score in metrics["precision_at_5"]:
        print(f"ID: {id}, Precision@5: {score:.4f}")

    print("\nAverage Precision@5:")
    for id, score in metrics["average_precision_at_5"]:
        print(f"ID: {id}, Average Precision@5: {score:.4f}")

    print("\nAccuracy@1:")
    for id, score in metrics["accuracy_at_1"]:
        print(f"ID: {id}, Accuracy@1: {score:.4f}")

    print("\nLatency:")
    for id, score in metrics["latency"]:
        print(f"ID: {id}, Latency: {score:.4f} seconds")
