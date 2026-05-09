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

# MuseTalk model paths
MODEL_CONFIG = {
    "musetalk": "/app/models/musetalk",
    "vae": "/app/models/sd-vae-ft-mse",
    "dwpose": "/app/models/dwpose",
    "face_parse": "/app/models/face-parse-bisent",
}

print("Loading MuseTalk pipeline...")

from musetalk.utils.utils import get_file_type, get_video_fps, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image

audio_processor, vae, unet, pe = load_all_model(
    unet_path=os.path.join(MODEL_CONFIG["musetalk"], "musetalkV15/unet.pth"),
    vae_type="sd-vae",
    unet_type="musetalkV15",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

print(f"MuseTalk loaded on {device}")

# Avatar cache — avoids reprocessing same image every call
avatar_cache = {}

def prepare_avatar(image_np, avatar_id):
    """Preprocess face region once per user, cache it."""
    if avatar_id in avatar_cache:
        return avatar_cache[avatar_id]

    coord_list, frame_list = get_landmark_and_bbox([image_np], 0)
    avatar_cache[avatar_id] = (coord_list, frame_list)
    return coord_list, frame_list


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

    # Decode audio to temp file
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    output_path = tempfile.mktemp(suffix=".mp4")

    try:
        # Get face coords (cached after first call per user)
        coord_list, frame_list = prepare_avatar(img_np, avatar_id)

        if not coord_list or coord_list[0] == coord_placeholder:
            return {"error": "No face detected in image"}

        # Process audio
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(
            feature_array=whisper_feature, fps=fps
        )

        # Generate lip-sync frames
        video_num = len(whisper_chunks)
        res_frame_list = []

        coord = coord_list[0]
        frame = frame_list[0]
        x1, y1, x2, y2 = coord

        for i, audio_feat in enumerate(whisper_chunks):
            audio_feat = torch.from_numpy(audio_feat).unsqueeze(0).to(device)

            ref_img = frame[y1:y2, x1:x2]
            ref_img_tensor = (
                torch.from_numpy(ref_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ).to(device)

            with torch.no_grad():
                latent = vae.encode(ref_img_tensor)
                pe_feat = pe(audio_feat)
                pred = unet(
                    latent,
                    timesteps,
                    encoder_hidden_states=pe_feat,
                ).sample
                pred_img = vae.decode(pred)

            pred_img_np = (
                pred_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)

            # Blend back onto full frame
            combined = get_image(frame, pred_img_np, coord)
            res_frame_list.append(combined)

        if not res_frame_list:
            return {"error": "No frames generated"}

        # Write video
        h, w = res_frame_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame_out in res_frame_list:
            writer.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))
        writer.release()

        # Mux audio into video
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