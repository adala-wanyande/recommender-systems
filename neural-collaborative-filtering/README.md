
This README guides users step-by-step to recreate the results, including all setup instructions, script explanations, and evaluation metrics.

---

# Neural Collaborative Filtering: Ablation Study Reproduction

---

## Prerequisites

- Python 3.7+
- pip
- CUDA-enabled GPU (optional but recommended for faster training)

---

## Setup

### Step 1: Install dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## Step 2: Data Preprocessing

This script prepares the dataset with the default negative sampling ratio of 4. It creates `train_df.csv`, `val_df.csv`, and `test_df.csv` under `./data/`.

```bash
python data_preprocessing.py
```

---

## Step 3: Train and Evaluate NCF Models (Embedding Ablation)

This script trains and evaluates multiple NCF model variants using different embedding dimensions and MLP layer sizes. Each configuration is saved under `./models/`.

```bash
python train_and_evaluate_ncf_models.py
```

You will see output after each model training, including:
- **NDCG@10**
- **Recall@10**

---

## Step 4: Negative Sampling Ablation

This performs a separate ablation study by varying the negative sampling ratio (e.g., 4, 8, 10, 15, 50, 100). It automatically creates new datasets and evaluates NCF models for each.

```bash
python evaluate_negative_sampling_ablation.py
```

For each ratio, results will be printed after training:
- **NDCG@10**
- **Recall@10**

---

## 🧠 Model Details

Models are based on [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031), using a fusion of GMF and MLP branches.

Key components:
- Dual embeddings for GMF and MLP
- Layered MLP with ReLU activations
- Sigmoid prediction layer
- Trained using binary cross-entropy

---

## 📁 Directory Structure

```text
.
├── data/
│   ├── train_df.csv
│   ├── val_df.csv
│   ├── test_df.csv
│   └── neg_ratio_*/  # Created dynamically
├── models/
│   └── *.pth          # Saved model checkpoints
├── data_preprocessing.py
├── train_and_evaluate_ncf_models.py
├── evaluate_negative_sampling_ablation.py
├── evaluate_ncf.py
├── neural_collaborative_filtering.py
└── requirements.txt
```

---