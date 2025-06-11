# evaluate_negative_sampling_ablation.py

import torch
import pandas as pd
from torch.utils.data import DataLoader
from neural_collaborative_filtering import NeuralCollaborativeFiltering, NCFDataset, train_model
from evaluate_ncf import evaluate_ranking_metrics
import os

# Negative sampling ratios tested
negative_ratios = [4, 8, 10, 15]

# Model settings (using the best-balanced configuration from previous tests)
embedding_dim = 64
mlp_layers = [128, 64, 32, 16]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Criterion and optimizer settings
criterion = torch.nn.BCELoss()

model_save_dir = './models/ablation'
os.makedirs(model_save_dir, exist_ok=True)

for ratio in negative_ratios:
    model_path = f'{model_save_dir}/ncf_neg_ratio_{ratio}.pth'

    if os.path.exists(model_path):
        print(f'Model for negative ratio 1:{ratio} already exists. Skipping training.')
        continue

    data_dir = f'./data/neg_ratio_{ratio}'

    print(f'\nTraining and evaluating with negative ratio: 1:{ratio}')

    train_df = pd.read_csv(f'{data_dir}/train_df.csv')
    val_df = pd.read_csv(f'{data_dir}/val_df.csv')
    test_df = pd.read_csv(f'{data_dir}/test_df.csv')

    num_users = max(train_df['user_id'].max(), val_df['user_id'].max(), test_df['user_id'].max()) + 1
    num_items = max(train_df['movie_id'].max(), val_df['movie_id'].max(), test_df['movie_id'].max()) + 1

    train_dataset = NCFDataset(train_df)
    val_dataset = NCFDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = NeuralCollaborativeFiltering(
        num_users,
        num_items,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=50,
        patience=5
    )

    # Evaluate the model
    evaluate_ranking_metrics(model, test_df, train_df, num_users, num_items, device, K=10)

    # Save model
    torch.save(model.state_dict(), model_path)