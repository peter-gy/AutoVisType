# Do Vision-Language Models See Visualizations Like Humans? Alignment in Chart Categorization

This repository implements the research described in our submission to the IEEE VIS 25 poster track. The project investigates whether state-of-the-art Vision-Language Models (VLMs) can align with human-centric, stimuli-based categorization of data visualizations.

## Overview

Unlike previous work that focuses on task-based data interpretation, this research probes whether VLMs can categorize visualizations based purely on their **essential visual stimuli** as perceived by human experts, independent of specific data interpretation tasks. We evaluate VLMs against [Chen et al.'s image-based typology](https://arxiv.org/abs/2403.05594) derived from expert analysis of the [VIS30K dataset](https://ieeexplore.ieee.org/abstract/document/9337213).

## Key Research Questions

- Can VLMs approximate human cognitive processes in visualization categorization?
- How well do VLMs grasp the "essential stimuli" that drive human expert categorization?
- What are the current limitations of strictly stimuli-based AI visual understanding in the visualization domain?

## Implementation

The research methodology is implemented using two interactive [Marimo](https://marimo.io/) notebooks.

### 📊 `src/inference.py`

Interactive notebook for running VLM inference on visualization images:

- Loads and samples the VIS30K dataset (stratified sampling of 305 images)
- Implements zero-shot categorization using structured prompts
- Supports concurrent processing with rate limiting and caching
- Outputs structured predictions for purpose, encoding, and dimensionality

### 📈 `src/evaluation.py`

Interactive notebook for comprehensive evaluation and analysis:

- Computes multi-label classification metrics (Accuracy, Hamming Loss, Jaccard Score, Precision/Recall/F1)
- Generates confusion matrices and performance visualizations
- Provides interactive exploration of results by model, feature, and difficulty
- Constructs comparative analysis across all evaluated models

## Evaluated Vision-Language Models (13 Total)

### Google GenAI

- `gemini-2.0-flash`
- `gemini-2.5-flash-preview-05-20`
- `gemini-2.5-pro-preview-05-06`

### OpenAI

- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `o4-mini`

### Meta LLaMA (via OpenRouter)

- `llama-4-scout`
- `llama-4-maverick`

### Mistral AI (via OpenRouter)

- `mistral-small-3.1-24b-instruct`
- `mistral-medium-3`
- `pixtral-large-2411`

### Qwen (via OpenRouter)

- `qwen2.5-vl-32b-instruct`

## Dataset & Evaluation Framework

- **Dataset**: [VIS30K](https://github.com/VisImageNavigator/VisImageNavigator.github.io/blob/54ab2319cca6a9e9056ce9cb5a337e920711b15e/public/dataset/vispubData30_updated_07112024.csv) with expert annotations (6,803 images)
- **Sample**: Stratified sample of 305 images across encoding types, dimensionalities, and difficulty levels
- **Features Evaluated**:
  - **Purpose**: `gui`, `schematic`, `vis`
  - **Encoding**: Various encoding types (bar, line, scatter, etc.)
  - **Dimensionality**: 2D, 3D, others
- **Setting**: Zero-shot evaluation with structured JSON output

## Key Findings

- **Purpose Identification**: VLMs achieve reasonable accuracy ($>0.7$) for high-level categorization
- **Dimensionality**: Performance varies with complexity, showing challenges with nuanced spatial reasoning
- **Encoding Recognition**: Most challenging task for all VLMs ($<0.4$ accuracy), highlighting the difficulty of discerning fine-grained visual stimuli
- **Difficulty Impact**: Performance decreases with expert-assessed image complexity across all models

## Getting Started

1. **Install Dependencies**:

```bash
uv sync
```

2. **Set up API Keys**:

Copy the example environment variables file and fill in the missing values:

```bash
cp .env.example .env
```

3. **Run Inference Notebook**:

```bash
uv run marimo run src/inference.py
```

4. **Run Evaluation Notebook**:

```bash
uv run marimo run src/evaluation.py
```

## Research Implications

This work is a precursor to a more comprehensive study that will provide insights for:

- **AI Development**: Understanding current VLM limitations in abstract visual reasoning
- **Human-AI Collaboration**: Informing the design of visualization tools that leverage human perceptual strengths
- **Visualization Research**: Establishing benchmarks for AI alignment with human-centric frameworks

## Future Work

- One-shot and few-shot prompting experiments
- Full VIS30K dataset evaluation
- Model uncertainty quantification
- Parameter sensitivity analysis
- Determinism evaluation across multiple runs
