from kagglehub import auth, kagglehub
import sys

path = kagglehub.dataset_download(sys.argv[1], output_dir=sys.argv[2])

print(f"Dataset downloaded to: {path}")
