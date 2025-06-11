import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data into a DataFrame
columns = ['user_id', 'movie_id', 'rating', 'timestamp']

df = pd.read_csv("ratings.dat", sep="::", engine="python", names=columns, encoding="ISO-8859-1") 

# Keep ratings >= 4 as implicit positive interactions
df_positive = df[df['rating'] >= 4].copy()
df_positive['interaction'] = 1  # positive interactions labeled as 1

# Drop unnecessary columns
df_positive = df_positive[['user_id', 'movie_id', 'interaction']]


def generate_negative_samples(df_positive, num_negatives=4):
    user_item_set = set(zip(df_positive['user_id'], df_positive['movie_id']))
    all_movie_ids = df_positive['movie_id'].unique()
    user_ids = df_positive['user_id'].unique()

    negative_samples = []

    all_movie_set = set(all_movie_ids)

    for user in user_ids:
        positive_items = df_positive[df_positive['user_id'] == user]['movie_id'].tolist()
        all_neg_items = list(all_movie_set - set(positive_items))

        num_neg_samples = min(len(all_neg_items), len(positive_items) * num_negatives)
        sampled_negatives = np.random.choice(all_neg_items, size=num_neg_samples, replace=False)

        for neg_item in sampled_negatives:
            negative_samples.append([user, neg_item, 0])

    return pd.DataFrame(negative_samples, columns=['user_id', 'movie_id', 'interaction'])


# Generate negatives (4 negatives per positive interaction)
df_negative = generate_negative_samples(df_positive, num_negatives=4)


# Positive interactions
df_positive_final = df_positive[['user_id', 'movie_id', 'interaction']]

# Combine data
df_full = pd.concat([df_positive_final, df_negative], ignore_index=True)

# Shuffle data
df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)


# First split: 70% training, 30% temp
train_df, temp_df = train_test_split(
    df_full, test_size=0.30, random_state=42, stratify=df_full['interaction'])

# Second split: 50% of temp (15% validation, 15% testing)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df['interaction'])

print(f"Train size: {len(train_df)} ({len(train_df)/len(df_full)*100:.2f}%)")
print(f"Validation size: {len(val_df)} ({len(val_df)/len(df_full)*100:.2f}%)")
print(f"Test size: {len(test_df)} ({len(test_df)/len(df_full)*100:.2f}%)")

train_df.to_csv('./data/train_df.csv', index=False)
val_df.to_csv('./data/val_df.csv', index=False)
test_df.to_csv('./data/test_df.csv', index=False)