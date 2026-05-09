import os
from huggingface_hub import snapshot_download

print("Downloading MuseTalk v1.5 models...")
snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    local_dir="/app/models/musetalk",
    ignore_patterns=["*.git*"]
)

print("Downloading face parsing model...")
snapshot_download(
    repo_id="jonathandinu/face-parsing",
    local_dir="/app/models/face-parse-bisent",
    ignore_patterns=["*.git*"]
)

print("Downloading DWPose model...")
snapshot_download(
    repo_id="yzd-v/DWPose",
    local_dir="/app/models/dwpose",
    ignore_patterns=["*.git*"]
)

print("Downloading SD VAE...")
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir="/app/models/sd-vae-ft-mse",
    ignore_patterns=["*.git*"]
)

print("All models downloaded.")