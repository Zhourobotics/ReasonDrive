# ReasonDrive

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2504.10757-b31b1b.svg)](https://arxiv.org/abs/2504.10757)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Models-ReasonDrive-orange)](https://huggingface.co/ac4462)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-DriveLM--reasoning-orange)](https://huggingface.co/datasets/ac4462/DriveLM-reasoning)

# ReasonDrive: Reasoning-Enhanced Small VLMs for Autonomous Driving

Official implementation of "ReasonDrive: Efficient Visual Question Answering for Autonomous Vehicles with Reasoning-Enhanced Small Vision-Language Models"

![Example Frame](assets/example_frame.png)

## Overview

ReasonDrive investigates whether explicitly modeling reasoning during fine-tuning enhances smaller, deployable Vision-Language Models (VLMs) on driving decision tasks. We introduce:

1. A reasoning-enhanced dataset derived from DriveLM using GPT-4o to generate structured reasoning chains for driving scenarios
2. Comprehensive evaluation of reasoning-based fine-tuning across multiple small VLM families
3. Evidence that explicit reasoning enhances internal representations for driving decisions

Our work shows that reasoning-enhanced fine-tuning creates efficient, interpretable models that address both computational and safety requirements for autonomous vehicles.

## Models

Our trained models are available on Hugging Face:

| Model | Description | Link |
|-------|-------------|------|
| Llama-3.2-11B-Vision-DriveLM-smpl | Llama 3.2 11B with standard fine-tuning | [ac4462/llama-3.2-11b-vision-DriveLM](https://huggingface.co/ac4462/llama-3.2-11b-vision-DriveLM) |
| Llama-3.2-11B-Vision-DriveLM-reason | Llama 3.2 11B with reasoning-enhanced fine-tuning | [ac4462/Llama-3.2-11B-Vision-DriveLM-Cot](https://huggingface.co/ac4462/Llama-3.2-11B-Vision-DriveLM-Cot) |
| Llava-1.5-7B-DriveLM-smpl | Llava 1.5 7B with standard fine-tuning | [ac4462/llava-1.5-7b-DriveLM](https://huggingface.co/ac4462/llava-1.5-7b-DriveLM) |
| Llava-1.5-7B-DriveLM-reason | Llava 1.5 7B with reasoning-enhanced fine-tuning | [ac4462/llava-1.5-7b-DriveLM-Cot](https://huggingface.co/ac4462/llava-1.5-7b-DriveLM-Cot) |
| Qwen2.5-VL-7B-DriveLM-smpl | Qwen 2.5 VL 7B with standard fine-tuning | [ac4462/Qwen2.5-VL-7B-DriveLM](https://huggingface.co/ac4462/Qwen2.5-VL-7B-DriveLM) |
| Qwen2.5-VL-7B-DriveLM-reason | Qwen 2.5 VL 7B with reasoning-enhanced fine-tuning | [ac4462/Qwen2.5-VL-7B-DriveLM-Cot](https://huggingface.co/ac4462/Qwen2.5-VL-7B-DriveLM-Cot) |
| Qwen2.5-VL-3B-DriveLM-smpl | Qwen 2.5 VL 3B with standard fine-tuning | [ac4462/Qwen2.5-VL-3B-DriveLM](https://huggingface.co/ac4462/Qwen2.5-VL-3B-DriveLM) |
| Qwen2.5-VL-3B-DriveLM-reason | Qwen 2.5 VL 3B with reasoning-enhanced fine-tuning | [ac4462/Qwen2.5-VL-3B-DriveLM-Cot](https://huggingface.co/ac4462/Qwen2.5-VL-3B-DriveLM-Cot) |

**Note:** While the model filenames on Hugging Face use "Cot" (Chain of Thought), we refer to these as "reason" models in our paper and documentation.

## Dataset

Our reasoning-enhanced dataset is available on Hugging Face:

- [ac4462/DriveLM-reasoning](https://huggingface.co/datasets/ac4462/DriveLM-reasoning)

This dataset enhances the original DriveLM benchmark with structured reasoning chains for driving scenarios, generated using GPT-4o.

Please refer to [DriveLM dataset](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge#how-to-prepare-data) to prepare the dataset.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ReasonDrive.git
cd ReasonDrive
```

## Docker Setup

We provide a Docker setup for reproducible environment:

```bash
# Pull the official Qwen Docker image
docker pull qwenllm/qwenvl:2.5-cu121

# Run Docker container with GPU support
./scripts/docker_run.sh
```

## Using Pre-trained Models

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image

# Load model and processor (example for Llama 3.2)
model_name = "ac4462/Llama-3.2-11B-Vision-DriveLM-Cot"  # Use the "reason" model
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Load camera views
camera_views = []
for cam in ["front_left", "front", "front_right", "back_left", "back", "back_right"]:
    img_path = f"path/to/CAM_{cam.upper()}.jpg"
    img = Image.open(img_path).convert("RGB")
    camera_views.append(img)

# Prepare prompt
question = "In this scenario, what are safe actions to take for the ego vehicle?"
prompt = f"Question: {question}\n"

# Process inputs and generate response
inputs = processor(text=prompt, images=camera_views, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)

print(response)
```

## Unsloth Fine-Tuning

We use [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning of our models. Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm-finetuning.py \
    --model_id "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit" \
    --output_dir "outputs_Qwen2.5-VL-3B-Instruct" \
    --epoch 5 \
    --batch_size 16 \
    --use_wandb \
    --wandb_project "ReasonDrive" \
    --wandb_run_name "qwen2.5-3B-reason"
```

## Model Evaluation

We follow a two-stage evaluation process:

### Stage 1: Model Response Generation

First, we generate model responses using the Unsloth framework:

```bash
python scripts/eval_finetuned_models.py \
    --model_path "outputs_Qwen2.5-VL-3B-Instruct/checkpoint-250" \
    --data_path "drivelm_with_thinking_800_frames_clean.json" \
    --base_image_path "/path/to/nuscenes" \
    --output_path "results/qwen2.5-3b-reason_responses.json" \
    --test_size 0.1 \
    --seed 3047
```

This script:
- Loads fine-tuned models using Unsloth
- Processes images and questions from the test split
- Generates responses with reasoning and answers
- Saves responses in JSON format for subsequent metric calculation

### Stage 2: Comprehensive Metrics Calculation

We then use the DriveLM evaluation framework to calculate multiple metrics:

```bash
python scripts/evaluation.py \
    --root_path1 "results/qwen2.5-3b-reason_responses.json" \
    --root_path2 "drivelm_with_thinking_800_frames_clean.json"
```

## Results

Our experiments demonstrate that reasoning-based fine-tuning consistently outperforms alternatives across multiple model families. The Llama3.2-11B-reason model achieves the highest overall performance while maintaining interpretability.

![Model Comparison](assets/model_comparison.png)



## Documentation

- [Using Models and Dataset](docs/model_usage.md)
- [Training Models](docs/training.md)
- [Unsloth Training](docs/unsloth_training.md)
- [Evaluation](docs/evaluation.md)

## Citation

```bibtex
@article{chahe2025reasondrive,
  title={ReasonDrive: Efficient Visual Question Answering for Autonomous Vehicles with Reasoning-Enhanced Small Vision-Language Models},
  author={Chahe, Amirhosein and Zhou, Lifeng},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This research was supported in part by NSF MRI Award Number 2320600.
- We thank the authors of DriveLM for releasing their dataset.
