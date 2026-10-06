# -*- coding: utf-8 -*-
import os
import cv2
import torch
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# === Setup BLIP-2 Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device)

# Prompt
blip2_prompt = (
    "Describe only the major physical actions or visible changes in body movement or object "
    "interaction happening in this frame. Exclude emotions, appearances, background details, "
    "and static poses unless they clearly indicate motion."
)

# === Function: Caption a Single Video ===
def caption_video(video_path, output_image_folder):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        print(f"?? Skipping {video_id}: FPS not detected")
        cap.release()
        return []

    duration_sec = int(total_frames / fps)
    results = []

    # Skip if less than 6 seconds
    if duration_sec < 6:
        print(f"?? Skipping {video_id}: too short ({duration_sec}s)")
        cap.release()
        return results

    # Subfolder for this video
    video_output_folder = os.path.join(output_image_folder, video_id)
    os.makedirs(video_output_folder, exist_ok=True)

    for sec in range(duration_sec):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if not success:
            continue

        # Save frame
        image_filename = f"{video_id}_{sec:02d}.jpg"
        image_path = os.path.join(video_output_folder, image_filename)
        cv2.imwrite(image_path, frame)

        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Generate caption
        inputs = processor(images=image, text=blip2_prompt, return_tensors="pt").to(device, torch.float16)
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        caption = processor.batch_decode(output, skip_special_tokens=True)[0].strip()

        results.append({
            "VideoName": video_id,
            "Timestamp": f"{sec:02d}s",
            "ImageFile": os.path.join(video_id, image_filename),
            "Caption": caption
        })

        print(f"[{video_id} - {sec}s] {caption}")

    cap.release()
    return results

# === Function: Caption Videos from CSV List ===
def caption_videos_from_csv(video_folder, list_file, output_csv, output_image_folder):
    # Load CSV and extract unique VideoNames
    df_list = pd.read_csv(list_file)
    video_list = df_list['VideoName'].unique().tolist()

    all_captions = []
    for video_name in video_list:
        # Match video file in folder
        candidates = [f for f in os.listdir(video_folder) if f.startswith(video_name)]
        if not candidates:
            print(f"?? No match found for: {video_name}")
            continue

        video_file = candidates[0]  # pick the first match
        video_path = os.path.join(video_folder, video_file)

        print(f"\n>> Processing video: {video_file}")
        captions = caption_video(video_path, output_image_folder)
        all_captions.extend(captions)

    # Save to CSV
    df_out = pd.DataFrame(all_captions)
    df_out.to_csv(output_csv, index=False)
    print(f"\n? Captions saved to: {output_csv}")
    print(f"??? Images saved in: {output_image_folder}")

# === Set Paths ===
video_folder = "/run/media/soumi/Devansh/LSMDC_Videos"
output_image_folder = "/run/media/soumi/Devansh/BLIP2_IMAGES"
list_file = "/run/media/soumi/Devansh/test_llava3_output.csv" # your CSV
output_csv = "/run/media/soumi/Devansh/test_blip2_output.csv"

os.makedirs(output_image_folder, exist_ok=True)

caption_videos_from_csv(video_folder, list_file, output_csv, output_image_folder)
