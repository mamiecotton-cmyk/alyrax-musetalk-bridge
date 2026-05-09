import runpod
import torch
import numpy as np
import cv2
import base64
import io
import os
import sys
import tempfile
import soundfile as sf
from PIL import Image

sys.path.insert(0, '/app/MuseTalk')

import insightface
from insightface.app import FaceAnalysis

from musetalk.utils.utils import load_all_model
from musetalk.utils.blending import get_image

# Load models
print("Loading face analysis...")
face_app = FaceAnalysis(providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(256, 256))

print("Loading MuseTalk pipeline...")
audio_processor, vae, unet, pe = load_all_model(
    unet_path="/app/models/musetalk/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_type="musetalkV15",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
print(f"MuseTalk loaded on {device}")

# Avatar cache
avatar_cache = {}

def get_face_bbox(image_np):
    faces = face_app.get(image_np)
    if not faces:
        return None
    face = faces[0]
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    # Add padding
    pad = 10
    h, w = image_np.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return (x1, y1, x2, y2)


def handler(job):
    job_input = job.get("input", {})

    image_b64 = job_input.get("image_base64", "")
    audio_b64 = job_input.get("audio_base64", "")
    avatar_id = job_input.get("avatar_id", "default")
    fps = job_input.get("fps", 25)

    if not image_b64 or not audio_b64:
        return {"error": "Missing image or audio"}

    # Decode image
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    # Decode audio
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    output_path = tempfile.mktemp(suffix=".mp4")

    try:
        # Get face bbox (cached)
        if avatar_id not in avatar_cache:
            bbox = get_face_bbox(img_np)
            if bbox is None:
                return {"error": "No face detected in image"}
            avatar_cache[avatar_id] = bbox
        
        x1, y1, x2, y2 = avatar_cache[avatar_id]

        # Process audio
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(
            feature_array=whisper_feature, fps=fps
        )

        if not whisper_chunks:
            return {"error": "No audio features extracted"}

        # Generate frames
        res_frame_list = []
        for audio_feat in whisper_chunks:
            audio_feat = torch.from_numpy(audio_feat).unsqueeze(0).to(device)

            face_region = img_np[y1:y2, x1:x2]
            face_tensor = (
                torch.from_numpy(face_region)
                .permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ).to(device)

            with torch.no_grad():
                latent = vae.encode(face_tensor)
                pe_feat = pe(audio_feat)
                pred = unet(latent, timesteps, encoder_hidden_states=pe_feat).sample
                pred_img = vae.decode(pred)

            pred_np = (
                pred_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)

            combined = get_image(img_np, pred_np, (x1, y1, x2, y2))
            res_frame_list.append(combined)

        if not res_frame_list:
            return {"error": "No frames generated"}

        # Write video
        h, w = res_frame_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in res_frame_list:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        # Mux audio
        muxed_path = output_path.replace(".mp4", "_muxed.mp4")
        os.system(
            f"ffmpeg -y -i {output_path} -i {audio_path} "
            f"-c:v copy -c:a aac -shortest {muxed_path} -loglevel error"
        )

        final_path = muxed_path if os.path.exists(muxed_path) else output_path

        with open(final_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        print(f"Generated {len(res_frame_list)} frames")
        return {
            "video_base64": video_b64,
            "fps": fps,
            "frames": len(res_frame_list),
        }

    finally:
        os.unlink(audio_path)
        for p in [output_path, output_path.replace(".mp4", "_muxed.mp4")]:
            if os.path.exists(p):
                os.unlink(p)


runpod.serverless.start({"handler": handler})