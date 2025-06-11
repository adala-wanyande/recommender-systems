# evaluate_ncf.py

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from neural_collaborative_filtering import NeuralCollaborativeFiltering, NCFDataset


def evaluate_ranking_metrics(model, test_df, train_df, num_users, num_items, device, K=10):
    model.eval()

    train_interactions = train_df.groupby(
        'user_id')['movie_id'].apply(set).to_dict()

    hits, ndcgs, recalls = [], [], []

    for user in test_df['user_id'].unique():
        positive_items = test_df[(test_df['user_id'] == user) & (
            test_df['interaction'] == 1)]['movie_id'].tolist()
        if not positive_items:
            continue

        negative_items = list(
            set(range(num_items)) - train_interactions.get(user, set()) - set(positive_items))
        negative_samples = np.random.choice(
            negative_items, 99, replace=False).tolist()

        items_to_rank = negative_samples + positive_items
        users_tensor = torch.tensor(
            [user] * len(items_to_rank), dtype=torch.long).to(device)
        items_tensor = torch.tensor(items_to_rank, dtype=torch.long).to(device)

        with torch.no_grad():
            predictions = model(users_tensor, items_tensor).cpu().numpy()

        item_scores = list(zip(items_to_rank, predictions))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_items = [x[0] for x in item_scores[:K]]

        hit = 0
        ndcg = 0
        num_hits = len(set(positive_items) & set(ranked_items))

        for pos_item in positive_items:
            if pos_item in ranked_items:
                hit = 1
                index = ranked_items.index(pos_item)
                ndcg = np.log(2) / np.log(index + 2)
                break

        recall = num_hits / len(positive_items)

        hits.append(hit)
        ndcgs.append(ndcg)
        recalls.append(recall)

    avg_hr = np.mean(hits)
    avg_ndcg = np.mean(ndcgs)
    avg_recall = np.mean(recalls)

    print(f"Hit Rate@{K}: {avg_hr:.4f}")
    print(f"NDCG@{K}: {avg_ndcg:.4f}")
    print(f"Recall@{K}: {avg_recall:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = pd.read_csv('./data/train_df.csv')
    test_df = pd.read_csv('./data/test_df.csv')

    num_users = max(train_df['user_id'].max(), test_df['user_id'].max()) + 1
    num_items = max(train_df['movie_id'].max(), test_df['movie_id'].max()) + 1

    model = NeuralCollaborativeFiltering(num_users, num_items).to(device)
    model.load_state_dict(torch.load('./models/best_ncf_model.pth'))

    evaluate_ranking_metrics(model, test_df, train_df,
                             num_users, num_items, device, K=10)
