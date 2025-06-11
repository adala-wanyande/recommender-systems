import torch
import pandas as pd
from torch.utils.data import DataLoader
from BERT4Rec_model import BERT4Rec
from train_and_evaluate import SequenceDataset, load_sequence_csv, train_model, build_item_popularity, evaluate_model

# ------------------ Hyperparameter Settings ------------------
sequence_lengths = [50, 100, 200]
mask_prob = 0.15
hidden_size = 128
num_layers = 4
batch_size = 64
num_epochs = 200
patience = 10

# ------------------ Load Data ------------------
train_df = load_sequence_csv('./Dataset/train.csv', column='Train')
val_df = load_sequence_csv('./Dataset/validation.csv', column='Validation')
test_df = load_sequence_csv('./Dataset/test.csv', column='Test')

train_sequences = train_df['Train'].tolist()
val_sequences = val_df['Validation'].tolist()
test_sequences = test_df['Test'].tolist()

num_items = max(max(train_df['Train'].explode()), max(
    val_df['Validation'].explode()), max(test_df['Test'].explode()))

item_counts = build_item_popularity(train_sequences)

# ------------------ Training and Evaluation Loop ------------------

results = []

for seq_len in sequence_lengths:
    print(f"\n=== Training with sequence length={seq_len} ===")

    # Pad or truncate sequences
    def pad_or_truncate(seqs, max_len):
        padded = []
        for seq in seqs:
            if len(seq) < max_len:
                padded_seq = [0] * (max_len - len(seq)) + seq
            else:
                padded_seq = seq[-max_len:]
            padded.append(padded_seq)
        return padded

    train_dataset = SequenceDataset(pad_or_truncate(train_sequences, seq_len))
    val_dataset = SequenceDataset(pad_or_truncate(val_sequences, seq_len))
    test_dataset = SequenceDataset(pad_or_truncate(test_sequences, seq_len))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = BERT4Rec(
        num_items=num_items,
        hidden_size=hidden_size,
        max_seq_length=seq_len,
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
        'sequence_length': seq_len,
        'recall@10': f"{recall:.4f}",
        'ndcg@10': f"{ndcg:.4f}"
    })

# ------------------ Save Summary ------------------
pd.DataFrame(results).to_csv(
    './ablation_results/ablation_sequence_length_results.csv', index=False)
