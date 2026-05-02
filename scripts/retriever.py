import requests
import argparse
import json
import pandas as pd
from pathlib import Path

import config


class Retriever:
    def __init__(self):
        self.index_df = pd.read_csv(config.INDEX_FILE)
        self.search_url = f"http://{config.API_HOST}:{config.API_PORT}/search"
        self.data_dir = config.WIKIART_DIR

    def search(self, image_path: str) -> list:
        image_path = Path(image_path)
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}
            response = requests.post(self.search_url, files=files)

        if response.status_code == 200:
            return self._file_paths(response.json()["ids"])
        else:
            return []

    def _file_paths(self, ids: list) -> list:

        paths = []
        for id in ids:
            filename = self.index_df[self.index_df["id"] == id]["filename"].values[0]
            paths.append(str(filename))

        return paths


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", "-i", type=str, help="Path to image file")

    parser.add_argument("--save", action="store_true")

    return parser.parse_args()


def retrieve():
    args = get_args()
    retriever = Retriever()

    if not args.save:

        image_path = args.image
        paths = retriever.search(image_path=image_path)

        print(*[f"{path}\n" for path in paths])

    else:
        query_df = pd.read_csv(config.QUERY_FILE)

        results = {}
        for _, row in query_df.iterrows():
            image_path = config.WIKIART_DIR / row["filename"]
            paths = retriever.search(image_path=image_path)
            results[row["id"]] = {
                "query": row["filename"],
                "result": paths,
            }

        with open("data/query_results/result.json", "w") as f:
            json.dump(results, f, indent=4)

        print("Results Saved to data/query_results/result.json")


if __name__ == "__main__":
    retrieve()
