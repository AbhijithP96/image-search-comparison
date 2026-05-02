# Art Image Retrieval System

## 1. Introduction

This project implements a content-based image retrieval (CBIR) system on the [WikiArt](https://www.wikiart.org/) dataset. Given a query image, the system retrieves the most visually similar artworks from an indexed collection.

Three retrieval strategies are implemented and evaluated:

- **Baseline** : Dense retrieval using DINOv2 (ViT-S) embeddings
- **Hybrid**: Concatenated DINOv2 + CLIP embeddings for a richer representation
- **Reranker** : Two-stage pipeline: DINO retrieval followed by CLIP-based reranking using CLIPScore

All embeddings are stored and queried via [Qdrant](https://qdrant.tech/) (local mode).

 > Report on the project, including the approach, trade-offs , improvement etc is documented in the [report](doc/Artwork%20Similarity%20Search%20Report%20.pdf).

---

## 2. Contents

- [Introduction](#1-introduction)
- [Folder Structure](#3-folder-structure)
- [Dataset Details](#4-dataset-details)
- [Setup Instructions](#5-setup-instructions)
- [Running the API](#6-running-the-api)
- [Query Images & Sample Outputs](#7-query-images--sample-outputs)
- [Strategy Comparison](#8-strategy-comparison-single-query-mode)
- [Multi-Image Aggregation](#9-multi-image-aggregation-experiment)
- [Latency Optimisation](#10-latency-optimisation)

---

## 3. Folder Structure

```
.
├── app.py
├── config.py             # Configuration constants (e.g. Qdrant collection names)
├── evaluator.py          # Evaluation script for retrieval metrics
├── requirements.txt
|── db/                   # Qdrant data directory (collections stored here)
|── docs/                 # Documentation and report
├── embedding/
│   ├── __init__.py
│   ├── baseline.py       # DINOv2 embedding + indexing
│   ├── clip.py           # CLIP embedding utilities
│   └── hybrid.py         # Hybrid (DINO + CLIP) embedding + indexing
├── search/
│   ├── __init__.py
│   ├── baseline.py       # BaselineSearcher
│   ├── hybrid.py         # HybridSearcher
│   ├── reranker.py       # RerankSearcher
│   └── utils.py          # Shared embedding utilities
├── data/
│   ├── visualize.py      # Genre distribution plots
│   ├── subset.py         # Index / query split creation
│   ├── kaggle_download.py
│   └── csv/
│       ├── index_set.csv
│       └── query_set.csv
└── scripts/
    ├── download_met.py   # Interactive Met Museum image downloader
    ├── multi_image_agg.py # Pooled embedding experiment
    └── retriever.py      # Standalone retrieval utility
```

---

## 4. Dataset Details

The WikiArt dataset is sourced from Kaggle. A balanced subset is constructed using `data/subset.py`:

- **Index set** : 200 images per genre (23 genres with ≥ 300 images only) from the training split, totalling 4600 images
- **Query set** : 1 image per genre, from 10 fixed styles from the test split

| Split | File | Images |
|-------|------|--------|
| Index | [`data/csv/index_set.csv`](data/csv/index_set.csv) | 4600 |
| Query | [`data/csv/query_set.csv`](data/csv/query_set.csv) | 10 |

Each CSV contains the columns: `id`, `filename`, `genre`, `artist`.

From the 4600 index images, only 4573 are successfully embedded and indexed due to some corrupted files. The 10 query images are all valid.

---

## 5. Setup Instructions

### Prerequisites

- Python 3.12
- CUDA-capable GPU recommended (CPU inference supported)

### Clone the repository

```bash
git clone https://github.com/AbhijithP96/image-search-comparison.git

cd image-search-comparison
```
### Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** requirements.txt includes torch and torchvision but does not specify a CUDA build. On Linux, pip installs a CUDA-enabled build by default. On Windows, it installs the CPU-only variant. For GPU support on Windows, install PyTorch manually from [here](https://pytorch.org/) for your CUDA version and remove torch and torchvison from requirements.txt before running the above command.

### Download the dataset

```bash
mkdir -p wikiArt

python data/kaggle_download.py steubk/wikiart wikiArt
```

### Build the index

Run each embedding script once to populate the Qdrant collections.

**Baseline (DINOv2):**
```bash
python -m embedding.baseline \
  --data-dir wikiArt \
  --index-file data/csv/index_set.csv
```

**Hybrid (DINOv2 + CLIP):**
```bash
python -m embedding.hybrid \
  --data-dir wikiArt \
  --index-file data/csv/index_set.csv
```

Both scripts write into `db/` under separate Qdrant collections (`baseline` and `hybrid`).

---

## 6. Running the API

```bash
python app.py
```

The server starts at `http://localhost:8000`.

### Endpoint

```
POST /search
Content-Type: multipart/form-data
Body: file=<image>
```

Returns a JSON response with the top-5 retrieved image IDs.

### Querying with `retriever.py`

A command-line client is provided at `scripts/retriever.py`. It posts an image to the running API and prints the filenames of the top-5 retrieved artworks:

```bash
python -m scripts.retriever -i wikiArt/Abstract_Expressionism/mark-tobey_washington.jpg
```

**Example output:**
```
Expressionism/pierre-alechinsky_untitled-from-the-portfolio-with-the-grain-of-the-wood-au-fil-du-bois-7-1973.jpg
 Abstract_Expressionism/jackson-pollock_moon-woman-1942(1).jpg
 Abstract_Expressionism/jackson-pollock_mural.jpg
 Abstract_Expressionism/jimmy-ernst_lookscape-1952.jpg
 Abstract_Expressionism/frank-stella_guadalupe-island-caracara-1979.jpg
```

The retriever resolves the raw IDs returned by the API back to image filenames using `data/csv/index_set.csv`.

---

## 7. Query Images & Sample Outputs

The 10 query images cover a diverse set of artistic styles:

To save the results of the entire query set, run the evaluation script:

```bash
python -m scripts.retriever --save
```

Ensure the api is running by executing `python app.py` before running the above command. This will save the retrieved results for all 10 query images into `data/query_results/result.json` as JSON files

Note: This has alredy been done and the [results](./data/query_results/result.json) are included in the repository for reference.

---

## 8. Strategy Comparison (Single Query Mode)

To run the evalution, ensure the qdrant collections are populated with the respective embeddings. Then execute:

```bash
python evaluator.py single
```

Evaluated over all 10 query images using `evaluator.py`.

| Method | P@5 | mAP@5 | Top-1 Acc | Latency (ms) | Notes |
|--------|-----|-------|-----------|--------------|-------|
| Baseline | 0.42 | 0.63 | 0.5 | 87.11 | DINOv2-small (384-d); fast single-stage retrieval |
| Hybrid | 0.54 | 0.71 | 0.5 | 78.99 | DINOv2 + CLIP concat (896-d); richer representation, higher indexing cost |
| Reranker | 0.56 | 0.67 | 0.5 | 613.94 | DINO retrieves top-10, CLIP reranks to top-5; highest quality, two-stage cost |

> Baseline & Hybrid: Hybrid achieves better P@5 and mAP@5 than Baseline, indicating that the CLIP features provide complimentary information to the DINOv2 embeddings, thereby improving retrieval quality. This might be beacuse CLIP captures conceptual information due to language supervision, while DINOv2 focuses on global visual semantics.

> Baseline & Reranker: Reranker achieves the best P@5, showing that CLIP-based reranking can effectively refine the initial DINO retreival results. However, the latency is significantly higher due to the two-stage process and the computational cost of CLIPScore.

> Top-1 Accuracy of all methods is the same, suggesting that hard queries are very visually similar to the retrieved top-1 result. This shows that a larger sample size for the index set is required or fine-tuning the model on the domain-specific dataset might be required.

---

## 9. Multi-Image Aggregation Experiment

Run via `scripts/multi_image_agg.py` on the [Met Museum](https://www.metmuseum.org/) collection (downloaded separately using `scripts/download_met.py`).

Note: Sampled Images are already included in the repository for reference.

Create a new Qdrant collection by running
the following command once:

```bash
python -m scripts.multi_image_agg create
```
Two collections are built and compared:

| Collection | Description |
|------------|-------------|
| `single` | One point per image, each photograph indexed independently |
| `pooled` | One point per artwork, embeddings from all photographs averaged (L2-normalised before mean pooling) |

Then run the following to perform retrieval and evaluation:

```bash
python -m scripts.multi_image_agg eval
```


Retrieval uses DINOv2-small embeddings. Evaluation label is artwork **class** (e.g. Paintings, Drawings). Metrics are computed over the Met query set.

| Method | Top-K | Top-1 Accuracy | P@5 |
|--------|-------|----------------|-----|
| Single |   2   |  0.75          | 0.75   |
| Single |   3   |  0.75          | 0.583  |
| Single |   5   |  0.75          | 0.40   |
| Pooled |   2   |  0.75          | 0.375  |
| Pooled |   3   |  0.75          | 0.25   |
| Pooled |   5   |  0.75          | 0.15   |

> The top1 accuracy of both methods is the same. This means the aggregated embedding successfully preserves the dominant visual characterisitics of the artwork. 

> The P@5 of the pooled model is significantly lower than the other. This is mainly because of the small sample size of the dataset rather than the pooling approach itself. The aggrgated embedding is one vector per artwork while single method has one vector per view. This effectively means the single method has a larger number of points to retrieve from. At smaller k, single method has larger pool of vectors to retrive from than the pooled method. At higher k, the pooled method has lesser vector. This effectively means bigger dataset for indexing and evaluation would give a better conclusion on the effectiveness of the pooling approach.

> The decrease in P@5 with increasing k is expected as the retrieved set is larger and more likely to contain irrelevant results.

---

## 10. Latency Optimisation

Batched Retrieval is implement to baseline searcher.

THe latency is compared with the original single-query searcher using `evaluator.py`:

```bash
python evaluator.py batch
```

| Method | Latency (ms) | 
|--------|--------------|
| Single-query | 63.03 |
| Batched (batch size=10) | 39.83 |

> The batched retrieval approach significantly reduces latency compared to the single-query method. By processing multiple queries together, we can leverage parallelism and reduce overhead, leading to faster retrieval times.

## Troubleshooting

- If you encounter errors like `ValueError: Collection pooled already exists`. Then kill the python process and rerun the script. This is because qdrant in local mode does not allow multiple clients to access the same collection at the same time.
- if you encounter `RuntimeError: Storage folder db is already accessed by another instance of Qdrant client.`. Then stop the running fastapi server and retry the command. 

