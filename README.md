# Vision-Language Temporal Action Conditioning Pipeline

A multi-stage vision-language data curation and action-conditioning
pipeline for constructing temporally aligned training data from video.

The pipeline extracts keyframes from videos, generates action-centric
visual captions using vision-language models, aligns past and future
frames through a sliding temporal window, and synthesizes
natural-language action directives for generative visual modeling.

## Overview

The implemented pipeline follows the structure:

``` text
Video Corpus
    │
    ▼
Frame Extraction & Captioning
(OpenCV + LLaVA-NeXT / BLIP-2)
    │
    ▼
Per-second Captions + Images
    │
    ▼
Temporal Windowing
(4-second past context + 1-second future target)
    │
    ▼
combined_windows.csv
    │
    ▼
Action Directive Synthesis
(Llama-3.1-8B-Instruct)
    │
    ▼
summarized_llama31.csv
```

The core conditioning structure is:

\[ I_t,; `\text{action}`{=tex},; I\_{t+1} \]

where the previous visual context and an action directive are used to
describe the subsequent visual state.

## Pipeline Components

### 1. Data Curation & Keyframe Extraction

The current implementation directly demonstrates the pipeline using the
**LSMDC (Large Scale Movie Description and Challenge)** video corpus.

The video data is accessed from:

``` text
/run/media/soumi/Devansh/LSMDC_Videos
```

Videos are processed with OpenCV using `cv2.VideoCapture`.

Key steps:

-   Videos shorter than 6 seconds are filtered out.
-   Frames are sampled at 1 FPS.
-   OpenCV timestamp seeking is used:

``` python
cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
```

-   Extracted images are organized into per-video directories.

Example output structure:

``` text
BLIP2_IMAGES/
└── <video_id>/
    ├── <video_id>_00.jpg
    ├── <video_id>_01.jpg
    ├── <video_id>_02.jpg
    └── ...
```

### 2. Frame-Level Action Captioning

Two vision-language models are used to generate dense, action-centric
descriptions:

  Script            Model
  ----------------- -------------------------------------
  `LLAVA_MAIN.py`   `llava-hf/llava-v1.6-mistral-7b-hf`
  `BLIP2.py`        `Salesforce/blip2-flan-t5-xl`

Both models are used in FP16 precision.

The captioning stage is deliberately constrained to focus on physical
dynamics rather than irrelevant visual details.

The prompting strategy asks the model to describe:

-   Major physical actions
-   Visible changes in body movement
-   Object interactions

and exclude:

-   Emotions
-   Appearance
-   Background details
-   Static poses unless they clearly indicate motion

The resulting annotations contain:

``` text
VideoID
Timestamp
ImageFile
Caption
```

Example output files referenced by the pipeline include:

``` text
test_llava3_output.csv
test_blip2_output.csv
```

## 3. Heuristic Filtering

The pipeline applies basic heuristics to improve caption informativeness
and temporal consistency.

### Duration filtering

Clips shorter than 6 seconds are removed:

``` text
duration_sec < 6
```

### Caption constraints

Negative prompting is used to discourage descriptions of:

-   Static backgrounds
-   Lighting
-   Emotional adjectives
-   Motionless poses

The goal is to preserve action-relevant information and reduce static
visual noise.

## 4. Temporal Window Construction

`dataset_merger.py` aligns the per-second captions into temporal
training examples.

For each example, the pipeline uses:

-   A **4-second rolling past context**
-   The **immediately following 1-second target state**

Conceptually:

\[ I\_{t-3}, I\_{t-2}, I\_{t-1}, I_t
`\quad`{=tex}`\longrightarrow`{=tex}`\quad`{=tex} I\_{t+1} \]

The temporal window therefore provides four frames of prior context
before the target frame.

The merger combines textual descriptions with absolute image locations,
producing records containing information such as:

``` text
[Image: {ImageFile}]
```

The resulting dataset is exported to:

``` text
combined_windows.csv
```

The implementation also references:

``` text
combined_windows_dataset_withimagelocationandname.csv
```

## 5. Action-Conditioned Directive Synthesis

`summarizer.py` converts the temporal context into a unified
natural-language action directive.

The summarization model is:

``` text
meta-llama/Llama-3.1-8B-Instruct
```

running in FP16.

The model receives:

-   The four-frame past context
-   The target future-frame description

The relevant conceptual structure is:

``` text
CONTEXT:
Past temporal observations

PRIORITY:
Target future-frame description
```

The model then produces an active-voice generative action prompt
conditioned on the preceding scene.

These prompts are intended to provide dense conditioning information
aligned with the target visual state for downstream diffusion or
autoregressive generation.

The final output is saved to:

``` text
summarized_llama31.csv
```

## Project Structure

The main scripts involved in the pipeline are:

``` text
.
├── BLIP2.py
├── LLAVA_MAIN.py
├── dataset_merger.py
├── summarizer.py
└── README.md
```

The principal responsibilities are:

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `BLIP2.py`                          Keyframe processing and
                                      BLIP-2-based visual captioning

  `LLAVA_MAIN.py`                     Keyframe processing and
                                      LLaVA-NeXT-based visual captioning

  `dataset_merger.py`                 Temporal alignment and construction
                                      of past/future windows

  `summarizer.py`                     Action directive synthesis using
                                      Llama-3.1-8B-Instruct
  -----------------------------------------------------------------------

## End-to-End Data Flow

``` text
                    LSMDC Video Corpus
                           │
                           ▼
                 ┌─────────────────────┐
                 │  OpenCV Frame       │
                 │  Extraction @ 1 FPS │
                 └──────────┬──────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ LLaVA-NeXT / BLIP-2   │
                │ Action-Centric Caption │
                └───────────┬────────────┘
                            │
                            ▼
                 Per-second captions
                    + image paths
                            │
                            ▼
                ┌────────────────────────┐
                │  dataset_merger.py    │
                │  4s context + 1s target│
                └───────────┬────────────┘
                            │
                            ▼
                  combined_windows.csv
                            │
                            ▼
                ┌────────────────────────┐
                │    summarizer.py       │
                │ Llama-3.1-8B-Instruct  │
                └───────────┬────────────┘
                            │
                            ▼
                  summarized_llama31.csv
```

## Supported Dataset Scope

The broader data-curation description references:

-   WebVid
-   LSMDC
-   ShareGemini

However, the provided implementation evidence directly demonstrates the
LSMDC pipeline. The LSMDC processing provides the direct evidence for
video frame extraction, keyframe generation, captioning, temporal window
construction, and triplet-oriented conditioning.

## Models

### LLaVA-NeXT

``` text
llava-hf/llava-v1.6-mistral-7b-hf
```

Used for frame-level visual description with an emphasis on physical
actions and visible changes.

### BLIP-2

``` text
Salesforce/blip2-flan-t5-xl
```

Used as a second vision-language captioning pipeline for dense
frame-level descriptions.

### Llama-3.1

``` text
meta-llama/Llama-3.1-8B-Instruct
```

Used to synthesize natural-language action directives from the temporal
context and target description.

## Outputs

The pipeline produces several intermediate and final artifacts:

``` text
test_llava3_output.csv
test_blip2_output.csv
combined_windows.csv
combined_windows_dataset_withimagelocationandname.csv
summarized_llama31.csv
```

These outputs represent progressively processed stages:

1.  Frame-level visual descriptions
2.  Temporally aligned context-target windows
3.  Synthesized action-conditioning directives

## Research Pipeline Objective

The central objective of the pipeline is to transform raw video into
temporally structured, action-aware conditioning data.

Rather than treating each frame independently, the pipeline explicitly
models the relationship between:

\[ `\text{Past visual context}`{=tex} `\rightarrow`{=tex}
`\text{Action directive}`{=tex} `\rightarrow`{=tex}
`\text{Future visual state}`{=tex} \]

This provides a structured representation suitable for downstream visual
generation experiments where an action-conditioned model must predict or
generate a future state from preceding visual observations.

## Implementation Notes

-   Frame extraction is performed at 1 FPS.
-   Clips shorter than 6 seconds are excluded.
-   Visual descriptions are generated using both LLaVA-NeXT and BLIP-2.
-   Captioning is constrained toward dynamic actions and object
    interactions.
-   Temporal alignment uses four preceding seconds and the immediately
    following second.
-   Action directives are synthesized using Llama-3.1-8B-Instruct.
-   The demonstrated execution path uses the LSMDC video corpus.

## Evidence / Implementation Summary

The repository's implementation corresponds to three major stages:

1.  **Data curation and triplet construction** --- keyframe extraction,
    captioning, and temporal pairing.
2.  **Heuristic filtering and caption informativeness** --- duration
    filtering and action-focused prompting.
3.  **Action-conditioned directive formulation** --- synthesis of
    natural-language directives using temporal context.

See the accompanying research documentation for the detailed
implementation evidence and workflow reconstruction.
