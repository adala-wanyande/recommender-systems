import pandas as pd
import numpy as np

# Load ratings.dat from MovieLens 1M dataset
column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings = pd.read_csv('./Dataset/ratings.dat', sep='::',
                      names=column_names, engine='python')

# Convert explicit ratings to implicit feedback (ratings >=4 => 1, otherwise drop)
ratings = ratings[ratings['Rating'] >= 4].copy()
ratings['Rating'] = 1  # set positive interactions as 1

# Sort by UserID and Timestamp
ratings.sort_values(by=['UserID', 'Timestamp'], inplace=True)

# Filter users with fewer than 5 interactions
interaction_counts = ratings['UserID'].value_counts()
valid_users = interaction_counts[interaction_counts >= 5].index
ratings = ratings[ratings['UserID'].isin(valid_users)].copy()

# Generate interaction sequences per user
user_groups = ratings.groupby('UserID')['MovieID'].apply(list)

# Sequence truncation/padding to fixed length (e.g., 20)
sequence_length = 100


def pad_or_truncate(sequence, maxlen=sequence_length):
    if len(sequence) >= maxlen:
        return sequence[-maxlen:]  # take latest interactions
    else:
        return [0] * (maxlen - len(sequence)) + sequence  # pad with zeros


user_sequences = user_groups.apply(lambda x: pad_or_truncate(x))

# Prepare sequences for splitting
sequences_df = pd.DataFrame({
    'UserID': user_sequences.index,
    'MovieSequence': user_sequences.values
})

# Split data per user into Train (70%), Validation (15%), and Test (15%)


def split_sequence(seq):
    total = len(seq)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)
    return seq[:train_end], seq[train_end:val_end], seq[val_end:]


sequences_df['Train'], sequences_df['Validation'], sequences_df['Test'] = zip(
    *sequences_df['MovieSequence'].apply(split_sequence))

# Display processed data
print("Sample processed data:")
print(sequences_df.head())

# Save train, validation, and test data to CSV files
sequences_df[['UserID', 'Train']].to_csv('./Dataset/train.csv', index=False)
sequences_df[['UserID', 'Validation']].to_csv(
    './Dataset/validation.csv', index=False)
sequences_df[['UserID', 'Test']].to_csv('./Dataset/test.csv', index=False)

print("CSV files saved: train.csv, validation.csv, test.csv")
