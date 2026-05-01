import requests
import argparse
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

    parser.add_argument(
        "--image", "-i", required=True, type=str, help="Path to image file"
    )

    return parser.parse_args()


def retrieve():
    args = get_args()

    image_path = args.image

    retriever = Retriever()

    paths = retriever.search(image_path=image_path)

    print(*[f"{path}\n" for path in paths])


if __name__ == "__main__":
    retrieve()
