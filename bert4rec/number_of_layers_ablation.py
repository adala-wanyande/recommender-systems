import torch
import pandas as pd
from torch.utils.data import DataLoader
from BERT4Rec_model import BERT4Rec
from train_and_evaluate import SequenceDataset, load_sequence_csv, train_model, build_item_popularity, evaluate_model

# ------------------ Hyperparameter Settings ------------------
num_layers_list = [2, 4, 6]
mask_prob = 0.15
hidden_size = 128
batch_size = 64
num_epochs = 200
patience = 10
max_seq_length = 100

# ------------------ Load Data ------------------
train_df = load_sequence_csv('./Dataset/train.csv', column='Train')
val_df = load_sequence_csv('./Dataset/validation.csv', column='Validation')
test_df = load_sequence_csv('./Dataset/test.csv', column='Test')

train_dataset = SequenceDataset(train_df['Train'].tolist())
val_dataset = SequenceDataset(val_df['Validation'].tolist())
test_dataset = SequenceDataset(test_df['Test'].tolist())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

num_items = max(max(train_df['Train'].explode()), max(
    val_df['Validation'].explode()), max(test_df['Test'].explode()))

item_counts = build_item_popularity(train_df['Train'].tolist())

# ------------------ Training and Evaluation Loop ------------------

results = []

for num_layers in num_layers_list:
    print(f"\n=== Training with num_layers={num_layers} ===")

    model = BERT4Rec(
        num_items=num_items,
        hidden_size=hidden_size,
        max_seq_length=max_seq_length,
        num_layers=num_layers,
        num_heads=4,
        dropout=0.2
    )

    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=1e-3,
        patience=patience,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    recall, ndcg = evaluate_model(
        trained_model,
        test_loader,
        item_counts,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    results.append({
        'num_layers': num_layers,
        'recall@10': f"{recall:.4f}",
        'ndcg@10': f"{ndcg:.4f}"
    })

# ------------------ Save Summary ------------------
pd.DataFrame(results).to_csv(
    './ablation_results/ablation_layers_results.csv', index=False)
