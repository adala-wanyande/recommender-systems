import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import random
from collections import Counter
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from BERT4Rec_model import BERT4Rec

# ------------------ Data Loading ------------------


def load_sequence_csv(path, column):
    df = pd.read_csv(path)
    df[column] = df[column].apply(ast.literal_eval)
    return df


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'input': torch.tensor(seq, dtype=torch.long)
        }

# ------------------ Utility Functions ------------------


def mask_input(input_seq, mask_token_id, mask_prob=0.15):
    input_seq = input_seq.clone()
    labels = input_seq.clone()
    mask = torch.rand(input_seq.shape) < mask_prob
    mask &= input_seq > 0
    input_seq[mask] = mask_token_id
    labels[~mask] = -100
    return input_seq, labels


def ndcg_k(ranked_items, ground_truth, k=10):
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in ground_truth:
            dcg += 1.0 / torch.log2(torch.tensor(i + 2.0))
    ideal_dcg = sum(1.0 / torch.log2(torch.tensor(i + 2.0))
                    for i in range(min(len(ground_truth), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_k(ranked_items, ground_truth, k=10):
    hits = sum([1 for item in ground_truth if item in ranked_items[:k]])
    return hits / len(ground_truth) if ground_truth else 0.0


def build_item_popularity(train_sequences):
    item_counts = Counter()
    for seq in train_sequences:
        item_counts.update(seq)
    return item_counts


def sample_popular_negatives(item_counts, all_items, num_samples, exclude_items=set()):
    sorted_items = [item for item, _ in item_counts.most_common()
                    if item not in exclude_items]
    return random.sample(sorted_items[:1000], num_samples)

# ------------------ Training ------------------


def train_model(model, train_loader, val_loader, num_epochs, lr=1e-3, patience=10, device='cuda'):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0, len(train_loader) * num_epochs)

    best_val_ndcg = 0
    patience_counter = 0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_seq = batch['input'].to(device)
            masked_input, labels = mask_input(input_seq, model.mask_token_id)
            optimizer.zero_grad()
            logits = model(masked_input)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_ndcgs = []
        with torch.no_grad():
            for batch in val_loader:
                input_seq = batch['input'].to(device)
                masked_input, labels = mask_input(
                    input_seq, model.mask_token_id)
                logits = model(masked_input)
                for i in range(logits.size(0)):
                    true_items = labels[i][labels[i] != -100]
                    if len(true_items) == 0:
                        continue
                    pred_scores = logits[i]
                    top_k_items = torch.topk(pred_scores, k=10, dim=-1).indices
                    ndcgs = [ndcg_k(top_k_items[j], [true_items[j].item()])
                             for j in range(len(true_items))]
                    val_ndcgs.extend(ndcgs)

        avg_val_ndcg = sum(val_ndcgs) / len(val_ndcgs)
        print(
            f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val NDCG@10: {avg_val_ndcg:.4f}')

        if avg_val_ndcg > best_val_ndcg:
            best_val_ndcg = avg_val_ndcg
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered!')
                break

    model.load_state_dict(torch.load('best_model.pt'))
    return model

# ------------------ Evaluation ------------------


def evaluate_model(model, test_loader, item_counts, num_negatives=100, device='cuda'):
    model.eval()
    model.to(device)
    recalls, ndcgs = [], []
    all_items = list(item_counts.keys())

    with torch.no_grad():
        for batch in test_loader:
            input_seq = batch['input'].to(device)
            masked_input, labels = mask_input(input_seq, model.mask_token_id)
            logits = model(masked_input)

            for i in range(logits.size(0)):
                true_items = labels[i][labels[i] != -100]
                if len(true_items) == 0:
                    continue
                for j in range(len(true_items)):
                    true_item = true_items[j].item()
                    negatives = sample_popular_negatives(
                        item_counts, all_items, num_samples=num_negatives, exclude_items=set(input_seq[i].tolist()))
                    candidates = [true_item] + negatives
                    scores = logits[i, j][candidates]
                    top_k = torch.topk(scores, k=10).indices.tolist()
                    ranked_items = [candidates[k] for k in top_k]
                    recalls.append(recall_k(ranked_items, [true_item]))
                    ndcgs.append(ndcg_k(ranked_items, [true_item]))

    avg_recall = sum(recalls) / len(recalls)
    avg_ndcg = sum(ndcgs) / len(ndcgs)
    print(
        f"Recall@10: {avg_recall:.4f}, NDCG@10: {avg_ndcg:.4f}")
    return avg_recall, avg_ndcg

# ------------------ Run Training and Evaluation ------------------


def run():
    train_df = load_sequence_csv('./Dataset/train.csv', column='Train')
    val_df = load_sequence_csv('./Dataset/validation.csv', column='Validation')
    test_df = load_sequence_csv('./Dataset/test.csv', column='Test')

    train_dataset = SequenceDataset(train_df['Train'].tolist())
    val_dataset = SequenceDataset(val_df['Validation'].tolist())
    test_dataset = SequenceDataset(test_df['Test'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    num_items = max(max(train_df['Train'].explode()), max(
        val_df['Validation'].explode()), max(test_df['Test'].explode()))

    model = BERT4Rec(num_items=num_items, hidden_size=128, num_layers=4)
    trained_model = train_model(model, train_loader, val_loader, num_epochs=200,
                                lr=1e-3, patience=10, device='cuda' if torch.cuda.is_available() else 'cpu')

    item_counts = build_item_popularity(train_df['Train'].tolist())
    evaluate_model(trained_model, test_loader, item_counts,
                   device='cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    run()
