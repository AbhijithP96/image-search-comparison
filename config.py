# config.py
import torch
from pathlib import Path

# Folder Paths
WIKIART_DIR = Path("wikiArt")
DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True, parents=True)

# Dataser CSV
INDEX_FILE = Path("data") / "csv" / "index_set.csv"
QUERY_FILE = Path("data") / "csv" / "query_set.csv"

# API url
API_HOST = "localhost"
API_PORT = 8000

# Model Name and Dimension
DINO_MODEL = "facebook/dinov2-small"
DINO_DIM = 384  # DINO V2 small has a feature dimension of 384
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_DIM = 512  # CLIP ViT-B/32 has a feature dimension of 512
HYBRID_DIM = DINO_DIM + CLIP_DIM

# device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collection Names
BASE = "baseline"
HYBRID = "hybrid"
POOLED = "POOLED"
