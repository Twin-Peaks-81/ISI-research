# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
csv_path = "/run/media/soumi/Devansh/combined_windows_nocap.csv"
output_csv = os.path.join(os.path.dirname(csv_path), "summarized_llama31.csv")

model_name = "meta-llama/Llama-3.1-8B-Instruct"  # gated model
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model & Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    use_auth_token=True
)

# --- Prompt Builder ---
def build_prompt(past, future):
    return (
        "You are a video scene summarizer.\n"
        "PRIORITY (focus): this action\n"
        "CONTEXT (fallback): past 4 frames description\n\n"
        f"CONTEXT:\n{past}\n\n"
        f"PRIORITY:\n{future}\n\n"
        "Write in one big paragraph ONLY in which the background details are with the help of the context, a this action image generative CAPTION\n"
        "(Use simple language, present tense, active voice, only allowed pronouns and articles)\n"
        "\nCAPTION:\n"
    )

# --- Summarization ---
def summarize(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only text after "CAPTION:"
    if "CAPTION:" in decoded:
        caption_text = decoded.split("CAPTION:", 1)[1].strip()
        caption_text = caption_text.split("\n\n")[0].strip()
        return caption_text
    return decoded.strip()

# --- Main Processing ---
df = pd.read_csv(csv_path)
summaries = []

for idx, row in df.iterrows():
    past = str(row["Dataset1_Captions"])
    future = str(row["Dataset2_Caption"])
    prompt = build_prompt(past, future)
    summ = summarize(prompt)
    summaries.append(summ)
    print(f"[{idx+1}/{len(df)}] CAPTION:", summ)

df["Summary"] = summaries
df.to_csv(output_csv, index=False)
print("Saved summarized output to", output_csv)
