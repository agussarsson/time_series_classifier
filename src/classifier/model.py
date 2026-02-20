import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score


class WindowTransformer(nn.Module):
    def __init__(self, lookback, n_features=3, d_model=32, nhead=8, nlayers=1, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.clf = nn.Linear(d_model, out_features=1)
        self.pos = nn.Parameter(torch.randn(1, lookback, d_model))

    def forward(self, x):
        x = self.in_proj(x)
        x = x + self.pos
        x = self.encoder(x)
        x = x.mean(dim=1)

        logits = self.clf(x).squeeze(-1)
        return logits
    


def make_model(hp, lookback, n_features):
    return WindowTransformer(
        lookback=lookback,
        n_features=n_features,
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        nlayers=hp["nlayers"],
        dropout=hp["dropout"],
    )