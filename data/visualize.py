import pandas as pd
import matplotlib.pyplot as plt
import config


def visualize_dataset_distribution():
    index_df = pd.read_csv(config.INDEX_FILE)
    query_df = pd.read_csv(config.QUERY_FILE)

    index_counts = index_df["genre"].value_counts()
    query_counts = query_df["genre"].value_counts()

    all_genres = sorted(set(index_counts.index) | set(query_counts.index))
    index_vals = [index_counts.get(g, 0) for g in all_genres]
    query_vals = [query_counts.get(g, 0) for g in all_genres]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Dataset Distribution  |  Index: {len(index_df)} images  |  Query: {len(query_df)} images",
        fontsize=13,
        fontweight="bold",
    )

    # --- genre distribution bar chart ---
    x = range(len(all_genres))
    width = 0.4
    ax = axes[0]
    ax.bar([i - width / 2 for i in x], index_vals, width=width, label="Index", color="steelblue")
    ax.bar([i + width / 2 for i in x], query_vals, width=width, label="Query", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_genres, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Images")
    ax.set_title("Genre Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- split size pie chart ---
    ax2 = axes[1]
    ax2.pie(
        [len(index_df), len(query_df)],
        labels=[f"Index\n({len(index_df)})", f"Query\n({len(query_df)})"],
        colors=["steelblue", "coral"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title("Index vs Query Split")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_dataset_distribution()
