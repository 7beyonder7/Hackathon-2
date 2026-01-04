# SDXL + CLIP Reranking Pipeline

A high-quality image generation pipeline that uses **Stable Diffusion XL** (or **SD 3.5 Large**) with **CLIP-based reranking** to automatically select the best generated image.

## 🎯 Overview

```
User Input → [Optional: LLM Prompt Conversion] → Generate N Candidates → CLIP Scoring → Best Image
```

The pipeline generates multiple image candidates and uses CLIP (Contrastive Language-Image Pre-training) to score how well each image matches your prompt, automatically selecting the best one.

---

## ✨ Features

- **Two Generation Modes:**
  - **Mode A (user_prompt):** Direct text-to-image with your prompt
  - **Mode B (auto_prompt):** LLM converts long text/articles into optimized image prompts

- **Smart Model Selection:**
  - `fast` / `quality_1min` presets → SDXL Base + Refiner
  - `quality_max` preset → SD 3.5 Large

- **CLIP Reranking:** Generates N candidates and picks the best match

- **Memory Management:** Auto-unloads models when switching between SDXL and SD3.5

- **Colab Optimized:** Includes fixes for Llama/vLLM compatibility

---

## 🚀 Quick Start

### 1. Set Your Preferences

```python
USE_VLLM = True           # Enable LLM for auto-prompt (Mode B)
USE_REFINER = True        # Use SDXL refiner for better details
PRESET = "quality_max"    # "fast" | "quality_1min" | "quality_max"
USE_TORCH_COMPILE = False # Enable for 10-30% speedup
```

### 2. Generate Images

**Mode A: Direct Prompt**
```python
result = generate_image(prompt="A golden retriever puppy playing in autumn leaves")
```

**Mode B: Auto-Prompt from Text**
```python
article = """
Climate scientists have observed unprecedented changes in Arctic ice patterns. 
Polar bears and Arctic foxes are among the species most affected...
"""
result = generate_image(mode="auto_prompt", text=article)
```

### 3. View All Candidates
```python
result = generate_image(
    prompt="A cozy coffee shop interior", 
    show=False, 
    return_candidates_images=True
)

# Display all candidates with their CLIP scores
for img, cand in zip(result["candidate_images"], result["candidates"]):
    print(f"Seed {cand['seed']}: CLIP={cand['clip']:.4f}")
```

---

## ⚙️ Configuration

### Presets

| Preset | Model | Resolution | Candidates | Speed | Quality |
|--------|-------|------------|------------|-------|---------|
| `fast` | SDXL | 768×768 | 2 | ~20s | Good |
| `quality_1min` | SDXL + Refiner | 1024×1024 | 2-4 | ~60s | Better |
| `quality_max` | SD 3.5 Large | 1024×1024 | 3-5 | ~90s | Best |

### Custom Configuration

```python
from dataclasses import dataclass

custom_cfg = ImgCfg(
    preset="custom",
    n_images=4,                # Number of candidates
    seed_base=42,              # Starting seed
    steps_base=40,             # Diffusion steps
    guidance=7.5,              # CFG scale
    clip_min=0.22,             # Minimum CLIP score threshold
    height=1024,
    width=1024,
    style_suffix=", cinematic lighting, 8k",  # Appended to prompts
    negative_prompt="blurry, low quality, watermark",
)

result = generate_image(prompt="Your prompt", cfg=custom_cfg)
```

---

## 📊 Understanding Output

### Result Dictionary

```python
{
    "ok": True,                    # Success/failure
    "best_seed": 42,               # Seed of best image
    "best_clip": 0.2585,           # CLIP score of best image
    "img_path": "images_out/...",  # Saved file path
    "timing_s": {
        "gen": 45.2,               # Generation time (seconds)
        "clip": 0.8                # CLIP scoring time
    },
    "candidates": [                # All candidates with scores
        {"seed": 42, "clip": 0.2585},
        {"seed": 43, "clip": 0.2341},
        ...
    ],
    "sd_prompt": "...",            # Final prompt sent to model
    "clip_prompt": "...",          # Prompt used for CLIP scoring
}
```

### CLIP Score Interpretation

| Score | Quality |
|-------|---------|
| < 0.20 | Poor - image doesn't match prompt well |
| 0.20 - 0.25 | Decent |
| 0.25 - 0.30 | Good ✓ |
| > 0.30 | Excellent ✓✓ |

### Seed Explained

- **Same seed + same prompt = identical image** (reproducible)
- Pipeline generates seeds: `[base_seed, base_seed+1, base_seed+2, ...]`
- Best seed is returned so you can regenerate that exact image

---

## 🔧 Troubleshooting

### Llama Fails to Load in Colab

The notebook includes automatic fixes, but if issues persist:

```python
# The fix adds these params for Llama models:
is_llama = "llama" in mid.lower()
extra_kwargs = {}
if is_llama:
    extra_kwargs["enforce_eager"] = True        # Disables CUDA graphs
    extra_kwargs["tensor_parallel_size"] = 1    # Single-GPU mode
```

### Out of Memory (OOM)

1. Use `PRESET = "fast"` (smaller resolution)
2. Set `USE_REFINER = False`
3. Reduce `cfg.n_images` to 2
4. The notebook auto-unloads models when switching between SDXL/SD3.5

### Low CLIP Scores

- Try different seeds: `generate_image(prompt="...", base_seed=100)`
- Increase candidates: modify `cfg.n_images`
- Improve your prompt with more specific details
- Lower the threshold: `cfg.clip_min = 0.18`

### JSON Serialization Error

Fixed in v2. The issue was logging PIL Images. Solution:
```python
# Log without images
log_rec = {...}  # no images
write_log(log_rec, cfg.log_jsonl)

# Return with images
result = log_rec.copy()
if return_candidates_images:
    result["candidate_images"] = ok_imgs
```

---

## 📁 Project Structure

```
├── images_out/              # Generated images
│   └── {id}_seed{N}_clip{score}.png
├── images_log.jsonl         # Generation logs (JSON lines)
└── Hackathon_2_main.ipynb   # Main notebook
```

---

## 🧠 How It Works

### 1. Prompt Processing
```
User prompt → Content filter → Add style suffix → Normalize
```

### 2. Image Generation
```
SDXL Base (N images) → [Optional: SDXL Refiner] → N candidate images
        OR
SD 3.5 Large (N images) → N candidate images
```

### 3. CLIP Reranking
```
For each image:
    text_embedding = CLIP.encode(prompt)
    image_embedding = CLIP.encode(image)
    score = cosine_similarity(text_embedding, image_embedding)

Select image with highest score
```

### 4. Quality Gate
```
if best_score < clip_min:
    return {"ok": False, "reason": "low_clip_alignment"}
else:
    save and return best image
```

---

## 📦 Dependencies

```
torch
diffusers
transformers
vllm (optional, for Mode B)
Pillow
```

---

## 🔗 Models Used

| Component | Model | Size |
|-----------|-------|------|
| SDXL Base | `stabilityai/stable-diffusion-xl-base-1.0` | ~6.5 GB |
| SDXL Refiner | `stabilityai/stable-diffusion-xl-refiner-1.0` | ~6.5 GB |
| SD 3.5 Large | `stabilityai/stable-diffusion-3.5-large` | ~16 GB |
| CLIP | `openai/clip-vit-large-patch14` | ~1.7 GB |
| LLM (optional) | `meta-llama/Llama-3.1-8B-Instruct` | ~16 GB |
| LLM (fallback) | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB |

---

## 📝 License

This project uses models with various licenses. Check individual model cards on Hugging Face for terms.

---

## 🙏 Acknowledgments

- [Stability AI](https://stability.ai/) - SDXL and SD 3.5
- [OpenAI](https://openai.com/) - CLIP
- [Meta](https://ai.meta.com/) - Llama
- [Alibaba](https://www.alibabacloud.com/) - Qwen
