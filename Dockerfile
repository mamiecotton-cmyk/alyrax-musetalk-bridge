FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models/hub

RUN apt-get update && apt-get install -y \
    ffmpeg libgl1-mesa-glx libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    runpod \
    torch==2.2.1 torchvision torchaudio \
    transformers==4.39.2 \
    diffusers==0.27.2 \
    accelerate==0.28.0 \
    omegaconf einops \
    opencv-python-headless \
    insightface onnxruntime-gpu \
    librosa soundfile \
    huggingface_hub \
    moviepy Pillow numpy

RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

COPY download_model.py .
RUN python download_model.py

COPY main.py .

CMD ["python", "-u", "main.py"]