# neural_collaborative_filtering.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Dataset Class
class NCFDataset(Dataset):
    def __init__(self, df):
        self.users = df['user_id'].values
        self.items = df['movie_id'].values
        self.labels = df['interaction'].values.astype('float32')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'item': torch.tensor(self.items[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# NCF Model
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_layers=[64,32,16,8]):
        super(NeuralCollaborativeFiltering, self).__init__()

        # GMF Embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP Embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP Layers
        mlp_input_dim = embedding_dim * 2
        mlp_modules = []
        for dim in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_input_dim, dim))
            mlp_modules.append(nn.ReLU())
            mlp_input_dim = dim
        self.mlp = nn.Sequential(*mlp_modules)

        # Fusion Layer
        self.fusion = nn.Linear(mlp_layers[-1] + embedding_dim, 1)

    def forward(self, user, item):
        # GMF Branch
        user_emb_gmf = self.user_embedding_gmf(user)
        item_emb_gmf = self.item_embedding_gmf(item)
        gmf_output = user_emb_gmf * item_emb_gmf

        # MLP Branch
        user_emb_mlp = self.user_embedding_mlp(user)
        item_emb_mlp = self.item_embedding_mlp(item)
        mlp_input = torch.cat((user_emb_mlp, item_emb_mlp), dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Fusion
        combined = torch.cat((gmf_output, mlp_output), dim=-1)
        logits = self.fusion(combined)
        prediction = torch.sigmoid(logits).squeeze()

        return prediction


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                users = batch['user'].to(device)
                items = batch['item'].to(device)
                labels = batch['label'].to(device)

                outputs = model(users, items)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), './models/best_ncf_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break


if __name__ == "__main__":
    # Load Data
    train_df = pd.read_csv('./data/train_df.csv')
    val_df = pd.read_csv('./data/val_df.csv')

    num_users = max(train_df['user_id'].max(), val_df['user_id'].max()) + 1
    num_items = max(train_df['movie_id'].max(), val_df['movie_id'].max()) + 1

    train_dataset = NCFDataset(train_df)
    val_dataset = NCFDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralCollaborativeFiltering(num_users, num_items).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5)
