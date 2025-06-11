# train_and_evaluate_ncf_models.py

import torch
import pandas as pd
from torch.utils.data import DataLoader
from neural_collaborative_filtering import NeuralCollaborativeFiltering, NCFDataset, train_model
from evaluate_ncf import evaluate_ranking_metrics
import os


# Ensure the models directory exists
os.makedirs('./models/', exist_ok=True)

# Define model configurations to test
model_configs = [
    {'embedding_dim': 32, 'mlp_layers': [64, 32, 16, 8]},
    {'embedding_dim': 64, 'mlp_layers': [128, 64, 32, 16]},
    {'embedding_dim': 128, 'mlp_layers': [256, 128, 64, 32]},
    {'embedding_dim': 24, 'mlp_layers': [48, 24, 12]}
]

# Load data
train_df = pd.read_csv('./data/train_df.csv')
val_df = pd.read_csv('./data/val_df.csv')
test_df = pd.read_csv('./data/test_df.csv')

num_users = max(train_df['user_id'].max(), val_df['user_id'].max(), test_df['user_id'].max()) + 1
num_items = max(train_df['movie_id'].max(), val_df['movie_id'].max(), test_df['movie_id'].max()) + 1

train_dataset = NCFDataset(train_df)
val_dataset = NCFDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Criterion and optimizer setup
criterion = torch.nn.BCELoss()

for config in model_configs:
    embedding_dim = config['embedding_dim']
    mlp_layers = config['mlp_layers']

    print(f"\nTraining model with embedding_dim={embedding_dim}, mlp_layers={mlp_layers}")

    # Initialize model
    model = NeuralCollaborativeFiltering(
        num_users,
        num_items,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
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

    # Save model
    model_filename = f"./models/ncf_emb{embedding_dim}_mlp{'-'.join(map(str, mlp_layers))}.pth"
    torch.save(model.state_dict(), model_filename)

# Evaluate all models
print("\nEvaluating all trained models:")
for config in model_configs:
    embedding_dim = config['embedding_dim']
    mlp_layers = config['mlp_layers']

    print(f"\nEvaluating model with embedding_dim={embedding_dim}, mlp_layers={mlp_layers}")

    model_filename = f"./models/ncf_emb{embedding_dim}_mlp{'-'.join(map(str, mlp_layers))}.pth"

    # Load model for evaluation
    model = NeuralCollaborativeFiltering(
        num_users,
        num_items,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers
    ).to(device)

    model.load_state_dict(torch.load(model_filename))

    # Evaluate
    evaluate_ranking_metrics(model, test_df, train_df, num_users, num_items, device, K=10)
