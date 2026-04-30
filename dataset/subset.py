import pandas as pd
from pathlib import Path
import ast

RANDOM_SEED = 42
# Random 10 styles to be included in the query set
QUERY_STYLES = [
    "Impressionism",
    "Cubism",
    "Baroque",
    "Abstract Expressionism",
    "Ukiyo e",
    "Art Nouveau Modern",
    "Minimalism",
    "Pop Art",
    "Northern Renaissance",
    "Pointillism",
]
IMAGES_PER_STYLE = 200
DATA_DIR = Path("wikiArt").absolute()
LABEL_FILE = DATA_DIR / "classes.csv"
OUTPUT_DIR = Path("dataset/csv").absolute()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUERY_FILE = OUTPUT_DIR / "query_set.csv"
INDEX_FILE = OUTPUT_DIR / "index_set.csv"


def create_query_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    To create the query set,
    this function filters the test subset of the dataset to include only the specified styles in QUERY_STYLES.
    Then, randomly samples one image from each of these styles to create a diverse query set.
    Finally, results are saved to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset information.
    Returns:
        pd.DataFrame: A DataFrame containing the query set with columns 'filename', 'genre', and 'artist'.
    """
    test_subset = df[df["subset"] == "test"]  # Filter the test subset of the dataset
    # print(test_subset["genre_str"].value_counts())

    # Filter to include only specified styles
    filtered = test_subset[test_subset["genre_str"].apply(lambda x: x in QUERY_STYLES)]

    # print(filtered["genre_str"].value_counts())

    # sample ONE images
    one_per_genre = filtered.groupby("genre_str", group_keys=False).apply(
        lambda x: x.sample(1, random_state=RANDOM_SEED)
    )

    one_per_genre = one_per_genre[["filename", "genre", "artist"]]

    print("Total number of images in the query set:", len(one_per_genre))

    return one_per_genre


def create_index_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    To create the index set,
    this function filters the training subset of the dataset to include styles that have at least 300 images.
    Then, it randomly samples 200 images from each of these styles to create a balanced index set.
    Finally, results are saved to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset information.
    Returns:
        pd.DataFrame: A DataFrame containing the index set with columns 'filename', 'genre', and 'artist'.
    """
    train_subset = df[df["subset"] == "train"]

    # Filter to include only styles with at least 300 images.
    filtered_indexes = (
        train_subset["genre_str"]
        .value_counts()[train_subset["genre_str"].value_counts() >= 300]
        .index
    )
    filtered = train_subset[train_subset["genre_str"].isin(filtered_indexes)]

    # print(filtered["genre_str"].value_counts())

    # Randomly sample 200 images from each of these styles to create a balanced index set.
    sampled_df = filtered.groupby("genre_str", group_keys=False).apply(
        lambda x: x.sample(200, random_state=RANDOM_SEED)
    )

    index_set = sampled_df[["filename", "genre", "artist"]]

    print("Total number of images in the index set:", len(index_set))

    return index_set


def save_dataframe_to_csv(df: pd.DataFrame, file_path: Path) -> None:
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        file_path (Path): The path where the CSV file will be saved.
    """
    df.index = range(1, len(df) + 1)
    df.to_csv(file_path, index=True, index_label="id")


if __name__ == "__main__":
    # Load the dataset from the  CSV file.
    df = pd.read_csv(LABEL_FILE)
    # Convert the 'genre' column from string representation of lists to actual lists,
    # and extract the first genre as a new column 'genre_str'.
    df["genre_str"] = df["genre"].apply(ast.literal_eval).apply(lambda x: x[0])

    index_set = create_index_set(df)
    index_set["genre"] = (
        index_set["genre"].apply(ast.literal_eval).apply(lambda x: x[0])
    )
    query_set = create_query_set(df)
    query_set["genre"] = (
        query_set["genre"].apply(ast.literal_eval).apply(lambda x: x[0])
    )

    # Save the query set and index set to CSV files.
    save_dataframe_to_csv(query_set, QUERY_FILE)
    print(f"Query set saved to {QUERY_FILE}")
    save_dataframe_to_csv(index_set, INDEX_FILE)
    print(f"Index set saved to {INDEX_FILE}")
