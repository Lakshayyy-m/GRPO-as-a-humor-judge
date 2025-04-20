# Project Environment and Setup Summary

## Development Environment Configuration Needed
- **OS Compatibility**: macOS (tested on MacBook Pro M2 Pro), Linux (Ubuntu 20.04+), and Google Colab
- **Python Version**: 3.10 recommended (tested with Python 3.13)
- **Memory Requirement**: Minimum 4 GB RAM (for inference); GPU (Nvidia T4) recommended for training
- **Virtual Environment**: Managed using Python `venv`

## Programming Languages
- **Python** – Main language for model training, inference, and interface logic
- **Markdown** – Used for project documentation (`README.md`, etc.)
- **Shell/Bash** – Used for scripting setup and running CLI commands

## Project Folder Hierarchy
```plaintext
project-root/
├── README.md
├── added_tokens.json
├── config.json
├── generation_config.json
├── merges.txt
├── model.safetensors
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```
- This structure represents a complete model deployment folder with tokenizer and generation configs.

## Installation Steps
```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install required libraries
pip install torch transformers safetensors gradio pandas unsloth
```

## Compile Commands (Execution/Inference)
``` 
# Run on Google Colab
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("RoyVoy/GRPO-as-a-humour-Judge")
model = AutoModelForCausalLM.from_pretrained("RoyVoy/GRPO-as-a-humour-Judge")

```

## Changes to Folder Structure After Compile
- If fine-tuning occurs, new files are saved:
  - `model.safetensors` (updated weights)
  - Tokenizer/config files may be regenerated if modified
- `__pycache__/` directories created by Python automatically
- Optional logs or outputs (e.g., `output_titles.txt`) may be added manually

## List of Tools Used and Corresponding Licenses
| Tool/Library               | Purpose                                  | License           |
|---------------------------|------------------------------------------|-------------------|
| Python                    | Core programming language                | PSF License       |
| PyTorch                   | Deep learning framework                  | BSD 3-Clause      |
| Hugging Face Transformers| Model/tokenizer handling                 | Apache 2.0        |
| Safetensors               | Safe model file format                   | Apache 2.0        |
| Unsloth                   | Efficient fine-tuning backend            | Apache 2.0        |
| Gradio                    | Web-based user interface                 | Apache 2.0        |
| Pandas                    | Dataset manipulation                     | BSD 3-Clause      |
| NumPy                     | Numerical utilities                      | BSD 3-Clause      |

## Team Task Log Summary
		
### Raghav

Led the training and evaluation of the model. Set up the fine-tuning pipeline, adjusted hyperparameters, and tested the model on validation data to ensure quality and relevance of generated titles.

### Amish

Worked on synthetic data generation to expand and diversify the training set. Generated sample abstracts and corresponding titles to help the model generalize better across domains.

### Lakshay

Conducted the initial research phase, exploring existing models, tools, and best practices for title generation. Helped define the system’s scope and evaluation criteria.

### Arsh

Handled data preprocessing, including cleaning the abstract-title pairs, formatting the dataset for model training, and ensuring consistency across inputs.






---
base_model: unsloth/Qwen2.5-1.5B-Instruct
tags:
- text-generation-inference
- transformers
- unsloth
- qwen2
- trl
- grpo
license: apache-2.0
language:
- en
---

# Uploaded  model

- **Developed by:** RoyVoy
- **License:** apache-2.0
- **Finetuned from model :** unsloth/Qwen2.5-1.5B-Instruct

This qwen2 model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)

# Git Repo
https://github.com/Lakshayyy-m/GRPO-as-a-humor-judge

