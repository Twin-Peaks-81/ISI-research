import os
import cv2
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# === Setup LLaVA-1.6-Mistral-7B-HF Model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LLaVA model and processor
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True # Add this for memory optimization
)
model.to(device)

# === Function: Caption a Single Video ===
def caption_video(video_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps)

    results = []

    # Define the new prompt as specified
    llava_prompt_template= "Describe only the major physical actions or visible changes in body movement or object interaction happening in this frame. Exclude emotions, appearances, background details, and static poses unless they clearly indicate motion. Focus strictly on what is actively changing or being done in this moment—such as gestures, turning, picking up, walking, etc.—and ignore minor shifts or ambiguous stillness."
    for sec in range(duration_sec): Here's the prompt, formatted as a single paragraph as you requested, that aims for exact and small outputs focused solely on activities: cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if not success:
            continue

        # Convert OpenCV image (BGR) to PIL image (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Generate caption using LLaVA's chat template
        # LLaVA models typically expect a specific conversational format.
        # The prompt includes <image> to indicate where the image should be inserted.
        
        # Constructing the conversation list as expected by LLaVA's processor
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": llava_prompt_template}]},
        ]
        
        # Apply the chat template to format the prompt correctly
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
        output = model.generate(**inputs, max_new_tokens=250, do_sample=False) # Increased max_new_tokens for detailed description

        # Decode the output, handling potential special tokens that LLaVA might generate
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        
        # Post-process the caption to remove the initial prompt if it's repeated
        # LLaVA might echo the prompt as part of its output. We need to clean this.
        # The output format for LLaVA is usually "USER: <image>\nPROMPT ASSISTANT: CAPTION"
        # We need to extract just the CAPTION part.
        assistant_prefix = "ASSISTANT:"
        if assistant_prefix in caption:
            caption = caption.split(assistant_prefix, 1)[1].strip()

        results.append({
            "VideoID": video_id,
            "Timestamp": f"{sec:02d}s",
            "Caption": caption
        })

        print(f"[{video_id} - {sec}s] {caption}")

    cap.release()
    return results

# === Function: Loop Over All Videos ===
def caption_folder(video_folder, output_csv):
    all_captions = []

    for file in os.listdir(video_folder):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(video_folder, file)
            print(f"\nðŸ“¹ Processing video: {file}")
            captions = caption_video(video_path)
            all_captions.extend(captions)

    # Save all to CSV
    df = pd.DataFrame(all_captions)
    df.to_csv(output_csv, index=False)
    print(f"\nâœ… All captions saved to: {output_csv}")

# === Set Your Folder and Output File ===
video_folder = "/run/media/soumi/Devansh/LSMDC_Videos"  # folder containing videos
output_csv = "/run/media/soumi/Devansh/test_llava3_output.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

caption_folder(video_folder, output_csv)
