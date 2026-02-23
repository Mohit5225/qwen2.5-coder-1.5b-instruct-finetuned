# Qwen2.5-Coder-1.5B-Instruct Fine-Tuning

This repository provides a comprehensive pipeline for fine-tuning the **Qwen2.5-Coder-1.5B-Instruct** model using **LoRA (Low-Rank Adaptation)**. The focus is on specialized instruction following and code generation, leveraging 4-bit quantization for efficient training on consumer-grade GPUs.

## 🚀 Key Features
- **Efficient Training**: Uses LoRA and 4-bit quantization (bitsandbytes) to significantly reduce VRAM requirements.
- **Dataset Preprocessing**: Tailored for JSON-based datasets containing instructions, constraints, and chain-of-thought examples.
- **Checkpointing**: Integrated support for saving and resuming training from checkpoints (configured for Google Drive).
- **Evaluation**: Includes scripts for qualitative evaluation and basic metric calculation on validation samples.
- **Model Merging**: Logic to merge LoRA adapters back into the base model for standalone deployment (e.g., with Ollama).

## 📁 Repository Structure
- `fineTuningbyQWEN3b.ipynb`: Main notebook containing the complete experimental fine-tuning logic.
- `fineTuningbyQWEN3b_cleaned.ipynb`: A streamlined and documented version of the pipeline.

## 🛠️ Getting Started

### Prerequisites
- A Google account for using **Google Colab**.
- Access to an NVIDIA GPU (T4 is sufficient for 4-bit training).
- A dataset in JSON format (formatted with instructions and expected responses).

### Steps to Run
1. Open `fineTuningbyQWEN3b_cleaned.ipynb` in Google Colab.
2. Mount your Google Drive to ensure your progress is saved.
3. Update the `DATASET_PATH` variable in the notebook to point to your JSON dataset.
4. Execute the cells sequentially to install dependencies, load the model, and begin the fine-tuning process.

## 📊 Training Configuration
- **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- **Technique**: LoRA (r=8, alpha=32)
- **Quantization**: 4-bit (NF4)
- **Optimizer**: Paged AdamW 8-bit
- **Scheduler**: Cosine learning rate decay

## 💾 Saving & Deployment
After training, the LoRA adapters are saved to your specified directory. The notebook also provides a cell to merge these adapters with the base model, creating a directory that can be used with `transformers` or converted for use in tools like `Ollama`.
