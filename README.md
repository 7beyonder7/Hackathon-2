# Developers Institute

## Generative AI Content Pipeline (A100 GPU)

End-to-end automated pipeline that **generates images from prompts**, runs **quality checks**, applies **ethical/safety filtering**, and saves clean outputs — optimized for an **NVIDIA A100**.

## Pipeline

`Prompt → Image Generator (Diffusion) → Quality Check → Safety Filter → Output (images + logs)`

## What’s inside

- **Image generation (GPU):** Stable Diffusion XL (SDXL) via `diffusers`
- **Quality control:** CLIP prompt–image similarity (reject low-alignment outputs)
- **Ethical/Safety filtering:** safety checker + prompt blocklist (and optional text toxicity model)
- **Automation:** run on-demand or scheduled (Python `schedule` or cron)
- **Evaluation:** generate 50–100 samples; track pass rate + score distributions

## Requirements

- Python 3.10+
- NVIDIA GPU + CUDA (A100 recommended)
- A Hugging Face account/token if the model requires it
