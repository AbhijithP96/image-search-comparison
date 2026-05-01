import requests
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
OBJECT_IDS = [
    197822,
    253370,
    500709,
    543901,
]


def get_data():

    metadata = {}

    for i, id in enumerate(OBJECT_IDS):
        response = requests.get(f"{URL}/{id}")
        if response.status_code == 200:
            data = response.json()
            metadata[i] = data

    with open("./met_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def select_images():
    met_dir = Path("met")
    met_dir.mkdir(exist_ok=True, parents=True)

    with open("./met_metadata.json", "r") as f:
        data = json.load(f)

    df_dict = {
        "id": [],
        "department": [],
        "title": [],
        "class": [],
        "filename": [],
        "subset": [],
    }
    idx = 0

    for id, met in data.items():
        print("Object: ", id)
        sub_dir = met_dir / str(met.get("objectID"))
        sub_dir.mkdir(exist_ok=True, parents=True)

        department = met.get("department")
        title = met.get("title")
        classification = met.get("classification")

        image_urls = [met.get("primaryImage"), met.get("primaryImageSmall")]

        image_urls.extend(met.get("additionalImages"))

        for url in image_urls:
            response = requests.get(url)
            image_buffer = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

            cv2.imshow("image", cv2.resize(image.copy(), (640, 640)))
            key = cv2.waitKey(0) & 0xFF

            if key == ord("s"):
                cv2.imwrite(str(sub_dir / f"{idx}_{title}.png"), image)
                df_dict["id"].append(idx)
                df_dict["class"].append(classification)
                df_dict["filename"].append(str(sub_dir / f"{id}_{title}.png"))
                df_dict["department"].append(department)
                df_dict["title"].append(title)
                df_dict["subset"].append("index")
                idx += 1
                print("image saved")
            if key == ord("d"):
                cv2.imwrite(str(sub_dir / f"{id}_{title}.png"), image)
                df_dict["id"].append(idx)
                df_dict["class"].append(classification)
                df_dict["filename"].append(str(sub_dir / f"{id}_{title}.png"))
                df_dict["department"].append(department)
                df_dict["title"].append(title)
                df_dict["subset"].append("query")
                idx += 1
                print("image saved")
            if key == ord("q"):
                print("image skipped")
                continue

    df = pd.DataFrame.from_dict(df_dict, orient="columns")
    df.to_csv(met_dir / "met.csv")


if __name__ == "__main__":
    get_data()
    select_images()
