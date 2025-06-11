# BERT4Rec MovieLens Recommendation - Assignment

## Overview
This project implements a BERT4Rec-based recommendation system on the MovieLens 1M dataset. It converts explicit ratings to implicit feedback, processes user sequences, trains a BERT4Rec model, and performs comprehensive evaluation including various ablation studies.

## Dataset
- **MovieLens 1M** dataset.
- Explicit ratings converted to binary implicit feedback:
  - Ratings >= 4 are treated as positive interactions.
  - Ratings < 4 are ignored.
- Users with fewer than 5 interactions are filtered out.
- Sequences are padded or truncated to a fixed maximum length.
- Dataset is split by user into:
  - 70% training
  - 15% validation
  - 15% testing

## Model Architecture
- **BERT4Rec** model based on Transformer encoder blocks.
- Components:
  - Item embeddings and positional embeddings.
  - Multi-head self-attention layers.
  - Feedforward layers.
  - Final output projection to item vocabulary size.
- Masked item prediction strategy (Masked Language Modeling objective).

## Training
- Optimizer: Adam with learning rate scheduling.
- Loss: Cross-entropy loss on masked positions only.
- Early stopping based on validation NDCG@10.
- Dropout and layer normalization are appropriately used.

## Evaluation
- Metrics:
  - Recall@10
  - NDCG@10
- Popularity-aware negative sampling is used during evaluation.
- Evaluation is performed only at masked positions.

## Ablation Studies
Conducted detailed ablation studies varying key hyperparameters:

- **Masking Ratio**: 0.15, 0.20, 0.25
- **Hidden Size**: 64, 128, 256
- **Number of Layers**: 2, 4, 6
- **Sequence Length**: 50, 100, 200
- **Number of Attention Heads**: 2, 4, 8

Each ablation study saves:
- Evaluation results (Recall@10, NDCG@10) in `.csv` files.

## Files
- `BERT4Rec_model.py` — BERT4Rec model implementation.
- `train_and_evaluate.py` — Training, evaluation, and data loading utilities.
- `*_ablation.py` — Ablation study scripts.

## How to Run
1. Preprocess and prepare `train.csv`, `validation.csv`, `test.csv`.
2. Train and evaluate model:
   ```bash
   python train_and_evaluate.py
   ```
3. Run ablation studies:
   ```bash
   python masking_ratio_ablation.py
   python hidden_size_ablation.py
   python number_of_layers_ablation.py
   python sequence_length_ablation.py
   python attention_heads_ablation.py
   ```
## Requirements
- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Pandas

## Author
- Ataklti Kidanemariam
- Benard Wanyande
