# Recommender Systems

![Thumbnail](./recommender_systems.png)

This repository is a collection of recommender systems projects. Each project resides in its own directory (`bert4rec`, `end-to-end-hybrid-pair-based`, `neural-collaborative-filtering`) and contains all the necessary code, data handling instructions, and documentation to reproduce the experimental results.

The aim is to showcase various recommender system techniques and provide clear, step-by-step guides within each project folder for replicating the documented findings.

## Repository Structure

```text
├── bert4rec/                      # BERT4Rec MovieLens Recommendation Project
├── end-to-end-hybrid-pair-based/  # Hybrid E-commerce Recommender Project
├── neural-collaborative-filtering/ # Neural Collaborative Filtering (NCF) Project
└── README.md                      # This file
```

## Projects Included

1.  **[BERT4Rec MovieLens Recommendation](./bert4rec/README.md)**
    *   Implements a sequential recommendation system on the MovieLens 1M dataset using BERT4Rec, including detailed ablation studies.
    *   **Reproduction:** See the `bert4rec/README.md` file for setup, data, and run instructions.

2.  **[Hybrid Recommendation System for E-commerce](./end-to-end-hybrid-pair-based/README.md)**
    *   A hybrid recommender system for the 'All Beauty' e-commerce category, combining user interactions and item metadata using pair-based models and blending.
    *   **Reproduction:** See the `end-to-end-hybrid-pair-based/README.md` file for the full pipeline (preprocessing, feature extraction, training, blending).

3.  **[Neural Collaborative Filtering (NCF) Ablation Study](./neural-collaborative-filtering/README.md)**
    *   An implementation of Neural Collaborative Filtering (NCF) with ablation studies on embedding dimensions, MLP structure, and negative sampling ratios.
    *   **Reproduction:** See the `neural-collaborative-filtering/README.md` file for setup, preprocessing, and how to run the ablation experiments.

## How to Reproduce Results

To reproduce the results for any specific project:

1.  Clone this repository.
2.  Navigate into the desired project's directory (e.g., `cd bert4rec`).
3.  Read the `README.md` file located inside that specific project directory.
4.  Follow the detailed instructions in the project's `README.md` to install dependencies, prepare data, and execute the code.

Each project's `README.md` is the definitive guide for its contents and result reproduction.

---

Explore the individual project directories to get started!
