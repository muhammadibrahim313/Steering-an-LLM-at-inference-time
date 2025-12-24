
# 🦙 Llama 3.1 Inference Steering with SAE Feature Vectors

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-PyTorch-orange)
![Library](https://img.shields.io/badge/Library-Transformers-yellow)
![Model](https://img.shields.io/badge/Model-Llama_3.1_8B_Instruct-blueviolet)

> **Control LLM behavior efficiently at inference time—no fine-tuning required.**

This project demonstrates **activation steering** using Sparse Autoencoder (SAE) feature vectors on **Llama 3.1 8B Instruct**. By injecting specific vectors into the model's residual stream during the forward pass, we can surgically alter the model's topic focus, persona, or stylistic output. This reproduces the core mechanism behind the famous [Hugging Face Eiffel Tower demo](https://huggingface.co/spaces/huggingface/eiffel-tower-llama-demo).

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Insights & Results](#-key-insights--results)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Work](#-future-work)

---

## 🔭 Project Overview

### The Problem
Traditional methods for modifying LLM behavior (RLHF, SFT) are computationally expensive and permanent. **Activation Engineering** offers a lightweight alternative: manipulating the internal state of the model *while it thinks*.

### The Solution
We use **SAE (Sparse Autoencoder)** features—specific directions in the model's high-dimensional space that correspond to interpretable concepts (e.g., "The Eiffel Tower", "Sadness", "Code"). By adding these vectors to the hidden states at specific layers, we "steer" the generation.

**Core Components:**
- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Technique**: PyTorch Forward Hooks
- **Data**: Pre-extracted SAE steering vectors

---

## 📊 Key Insights & Results

This project highlights the power of **inference-time intervention**. By sweeping through different steering scales, we observe how the model's output shifts from its baseline training to the steered concept.

### 1. The Steering Mechanism
The code introduces a "clamp" mechanism to ensure the steering vector is applied consistently without exploding the activation magnitudes.

| Parameter | Description | Effect |
| :--- | :--- | :--- |
| **Scale** | Scalar multiplier (0.0 - 1.0+) | Controls the *intensity* of the injected concept. |
| **Clamp** | Boolean | Prevents the feature from growing too large, maintaining coherence. |
| **Layer** | Integer (0-31) | Determines *where* in the reasoning process the concept is injected. |

### 2. Baseline vs. Steered Comparison
*Note: Results depend on the specific vectors loaded.*

| **State** | **Prompt** | **Output Behavior** |
| :--- | :--- | :--- |
| **Baseline** 🔵 | "Who are you?" | Standard, helpful assistant response. ("I'm an artificial intelligence...") |
| **Steered** 🔴 | "Who are you?" | **Subtle Shift**: At lower scales (0.5), the model maintains fluency but shifts tone. |
| **Over-Steered** ⚠️ | "Who are you?" | **Derailment**: At high scales (>1.5), the model becomes repetitive or nonsensical. |

### 3. The "Sweet Spot"
The analysis confirms that activation steering requires finding a **"sweet spot"** (typically scale ~1.0).
- **Too Low**: No noticeable effect.
- **Too High**: The model hallucinated or breaks grammar.
- **Just Right**: The model adopts the target concept while remaining grammatically correct.

---

## ⚙️ Methodology

The project follows a linear pipeline:

1.  **Initialization**:
    -   Securely load the Hugging Face token.
    -   Initialize Llama 3.1 8B Instruct with `torch.float16` for memory efficiency.
2.  **Vector Loading**:
    -   Download `steering_vectors.pt`.
    -   Load specific feature vectors identified by `[layer, feature_id, strength]`.
3.  **Hook Implementation**:
    -   Define `create_steering_hook`. This function intercepts the model's `hidden_states`.
    -   Mathematically add the steering vector: $h_{new} = h_{old} + (\text{strength} \times \text{vector})$.
4.  **Generation Loop**:
    -   Run a baseline generation (Temperature 0).
    -   Register hooks.
    -   Run steered generations at varying scales (0.5, 0.8, 0.9).
    -   Remove hooks to reset the model.

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- GPU recommended (Runs on Colab T4 or similar)
- Hugging Face Access Token (for gated Llama 3.1 models)

### Dependencies
Install the required libraries:

```bash
pip install -U transformers accelerate huggingface-hub safetensors sentencepiece torch
```

### Authentication
You must export your HF token or input it interactively:

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

---

## 🚀 Usage

### 1. Configuration
The `cfg` dictionary controls the steering parameters. You can modify the `features` list to target different layers or concepts if you have the feature IDs.

```python
cfg = {
    "llm_name": "meta-llama/Llama-3.1-8B-Instruct",
    "features": [
        [11, 74457, 0.128], # [Layer, FeatureID, Strength]
        [19, 93,    0.459],
        # ... add more features
    ],
    "max_new_tokens": 256
}
```

### 2. Running Inference
The core logic resides in the `generate_chat` function. To experiment with steering strength:

```python
# Steer with scale 0.8
output = generate_chat(
    chat=[{"role": "user", "content": "Tell me a story."}],
    steer=True,
    scale=0.8,
    clamp_intensity=True
)
print(output)
```

---

## 🔮 Conclusion & Future Work

This notebook successfully reproduces the **SAE steering workflow**. It proves that we can modify the behavior of a 70B+ parameter equivalent quality model (Llama 3 8B) simply by adding vectors at runtime.

**Future Directions:**
-   **Negative Steering**: Subtracting vectors to *suppress* concepts (e.g., removing refusal mechanisms).
-   **Dynamic Steering**: Adjusting the scale dynamically based on the input prompt.
-   **Feature Discovery**: Using [Neuronpedia](https://neuronpedia.org/) to find vectors for specific emotions or writing styles and plugging them into this pipeline.

---
*Created by Muhammad Ibrahim. Follow on [Github](https://github.com/muhammadibrahim313) and [Kaggle](https://www.kaggle.com/ibrahimqasimi).*
