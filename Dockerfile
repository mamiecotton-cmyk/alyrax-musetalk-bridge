RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    runpod \
    torch==2.2.1 torchvision torchaudio \
    "huggingface_hub==0.21.4" \
    transformers==4.39.2 \
    diffusers==0.27.2 \
    accelerate==0.28.0 \
    omegaconf einops \
    opencv-python-headless \
    insightface onnxruntime-gpu \
    librosa soundfile \
    moviepy Pillow numpy