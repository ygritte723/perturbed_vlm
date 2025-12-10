# Enhancing Biomedical Multi-modal Representation Learning

## Overview

This repository hosts the implementation of a novel approach to enhance **Vision-Language Models (VLMs)** for biomedical applications. By introducing **text perturbations as negative samples** and utilizing **localized attention mechanisms**, our model achieves a fine-grained understanding of medical imagery and associated reports.

![Architecture Diagram](assets/Per_arch.png)

## Methodology

### Approach

Our architecture integrates separately pretrained unimodal feature extractors for images and text. It uses a contrastive projection strategy to fuse cross-modal embeddings into a shared joint space. Key innovations include:

- **Local Attentive Contrastive Loss**: Improves precision in medical Image-Text matching by aligning specific image sub-regions with relevant text fragments.
- **Report Perturbation Sensitivity Loss**: Enhances the understanding of clinical semantics by focusing on sentence structure and parts of speech, complementing the standard Image-Report Matching Contrastive Loss.

### Text Perturbation Methods

To train the model to discriminate between original and perturbed reports, we employ several text manipulation techniques to generate negative samples:

| Perturbation Type | Description |
|-------------------|-------------|
| **Shuffle All Words** | Randomly shuffles all words in a sentence. |
| **Swap Adjacent Words** | Swaps adjacent words in the sentence. |
| **Reverse Sentence** | Reverses the order of words in a sentence. |
| **Shuffle Within Trigrams** | Shuffles words within each trigram in the sentence. |
| **Shuffle Trigrams** | Shuffles the trigrams within the sentence. |
| **Shuffle Nouns and Adjectives** | Shuffles nouns and adjectives while keeping other words fixed. |
| **Shuffle All but Nouns and Adjectives** | Shuffles all parts of speech except nouns and adjectives. |
| **Shuffle Nouns, Verbs, and Adjectives** | Shuffles nouns, verbs, and adjectives only. |
| **Replace Adjectives with Antonyms** | Replaces adjectives in the sentence with their antonyms. |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ygritte723/perturbed_vlm.git
    cd perturbed_vlm
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration:**
    Update `config.py` to point to your local dataset paths and checkpoint directories.
    ```python
    # Example in config.py
    IMAGES_ROOT = "/path/to/your/images"
    ...
    ```

## Project Structure

```text
perturbed_vlm/
├── config.py             # Configuration for file paths
├── evaluation/           # Scripts for evaluating the model
├── health_multimodal/    # Core package containing model definitions
│   ├── image/            # Image encoding and processing
│   └── text/             # Text encoding and processing
├── scripts/              # Training and utility scripts
│   ├── our.py            # Main training script for the proposed model
│   └── ...
└── requirements.txt      # Python dependencies
```

## Dataset

The model is trained on a curated dataset of biomedical images and associated reports.
- **Open-I**: Chest X-ray dataset with radiology reports.
- **RadNLI and MedNLI**: Benchmarks containing labelled hypothesis and premise pairs.
- **CheXpert**: Chest radiographs with associated radiology reports.

## Evaluation

Our model demonstrates significant improvements across various biomedical vision-language tasks.

### 1. Fine-Tuned Multi-task Image Classification (CheXpert)

| Model | Consolidation (%) | Pleural Effusion (%) | Mean Accuracy (%) |
|-------|-------------------|----------------------|-------------------|
| CLIP [3] | 28.80 | 43.60 | 36.20 |
| GLoRIA [12] | 71.11 | 28.89 | 50.00 |
| Ours (w/o local loss) | 33.80 | 77.80 | 55.80 |
| **Ours** | **93.40** | **51.80** | **72.60** |

### 2. Fine-Tuned Text Classification (RadNLI & MedNLI)

| Model | MedNLI Accuracy (%) | RadNLI Accuracy (%) |
|-------|---------------------|---------------------|
| CLIP [3] | 86.80 | 68.50 |
| GLoRIA [12] | 86.64 | 68.33 |
| Ours (w/o local loss) | 87.62 | 66.67 |
| **Ours** | 85.79 | **68.96** |

### 3. Zero-Shot Clinical Semantic Structure Evaluation (Open-I)

| Model | Open-I Accuracy (%) |
|-------|---------------------|
| CLIP [3] | 43.10 |
| GLoRIA [12] | 44.30 |
| Ours (w/o local loss) | 46.30 |
| **Ours** | **49.00** |