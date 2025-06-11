import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_size=256, max_seq_length=200, num_heads=4, num_layers=4, dropout=0.2):
        super(BERT4Rec, self).__init__()

        self.mask_token_id = num_items + 1

        config = BertConfig(
            vocab_size=num_items + 2,  # +1 for padding, +1 for mask token
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_seq_length,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, num_items + 1)

    def forward(self, input_seq):
        attention_mask = (input_seq > 0).float()
        embeddings = self.bert(input_ids=input_seq,
                               attention_mask=attention_mask).last_hidden_state
        embeddings = self.dropout(embeddings)
        logits = self.out(embeddings)
        return logits
