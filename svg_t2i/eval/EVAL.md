# SVG-T2I Evaluation

Use the `sample_svg_output.sh` script to generate images for benchmark prompts, including:
- **DPG-Bench**
- **GenEval**

## Environment Setup

### Installation

Please refer to the installation guides of the original repositories:

1. **DPG-Bench**: https://github.com/TencentQQGYLab/ELLA/tree/main  
2. **GenEval**: https://github.com/djghosh13/geneval

## Available Prompt Datasets

The `prompts/` directory provides the following datasets:

### 1. DPG-Bench (`dpg.jsonl`)

### 2. GenEval
- **`geneval_metadata.jsonl`**: Original GenEval text prompts  
- **`geneval_metadata_long.jsonl`**: Refined and extended GenEval prompts

## Usage

### Basic Usage

1. **Edit Configuration**: Open the `sample_svg_output.sh` script and modify the following settings:

```bash
# Select the prompt file
PROMPT_FILE="prompts/dpg.jsonl"              # DPG-Bench
# PROMPT_FILE="prompts/geneval_metadata.jsonl"      # GenEval (original)
# PROMPT_FILE="prompts/geneval_metadata_long.jsonl" # GenEval (enhanced)

# Set the pretrained model path
CKPT="pretrained_models/model.ckpt"

# Configure GPU devices
GPU_LIST="0,1,2,3,4,5,6,7"

# Set output resolution
RES="1024:1024x1024"
```

2. **Run the Script**:

```bash
bash eval/sample_svg_output.sh
```
3. **Compute Metrics**：
Follow the evaluation code from the original repositories to compute the corresponding metrics.
