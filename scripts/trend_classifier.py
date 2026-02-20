import torch
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import torch.nn as nn
import json
from classifier.config import HORIZON, LOOKBACK, DEVICE, STRIDER, TEST_SIZE, N_FEATURES, TRAIN_KWARGS
from classifier.data_processing import run_processing, trim_set
from classifier.model import make_model
from classifier.train import train_one_fold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from classifier.cv_tuning import tune_hparams_ts_cv

print(f"Running on {DEVICE}.")

X_trval, y_trval, X_test, y_test, idxs_trval = run_processing("AAPL", LOOKBACK, HORIZON, STRIDER, TEST_SIZE)


best_hp = {}
with open("./search_results/cv_search_results.json", "r") as f:
    best_hp = json.load(f)

final_model = make_model(best_hp, LOOKBACK, N_FEATURES)

tscv = TimeSeriesSplit(n_splits=5)
train_idx, val_idx = list(tscv.split(X_trval))[-1]

train_idx = trim_set(train_idx, val_idx, idxs_trval, HORIZON)

X_tr, y_tr = X_trval[train_idx], y_trval[train_idx]
X_val, y_val = X_trval[val_idx], y_trval[val_idx]

mean = X_tr.mean(dim=(0,1), keepdims=True)
std  = X_tr.std(dim=(0,1), keepdims=True) + 1e-8

X_tr  = ((X_tr  - mean) / std).to(DEVICE)
X_val = ((X_val - mean) / std).to(DEVICE)
X_test = ((X_test - mean) / std).to(DEVICE)

final_metrics = train_one_fold(
    final_model, X_tr, y_tr, X_val, y_val,
    device=DEVICE, **TRAIN_KWARGS
)

final_model.load_state_dict(final_metrics["best_state"])
final_model.eval()

torch.save({
    "state_dict": final_model.state_dict(),
    "hyperparams": best_hp,
    "mean": mean,
    "std": std
}, "./models/final_model.pt")

print("Final model saved.")

# Load best performing model + find best probability threshold based on val data.

with torch.no_grad():
    probs_val = torch.sigmoid(final_model(X_val))

y_val_i = y_val.long()

best_t, best_f1 = 0.5, -1.0
for t in np.linspace(0.0, 1.0, 101):
    preds = (probs_val > t).long()
    f1 = f1_score(
        y_val_i.cpu().numpy(),
        preds.cpu().numpy(),
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_t = t


print("best val acc/th:", best_f1, best_t)

# Apply the found threshold to the test set.

with torch.no_grad():
    probs_test = torch.sigmoid(final_model(X_test))

preds = (probs_test > best_t).long()

test_acc = (preds == y_test.long().to(DEVICE)).float().mean().item()
print(f"Test accuracy: {test_acc*100:.2f}%")

plt.plot(final_metrics["train_losses"], label="Train-Loss")
plt.plot(final_metrics["val_losses"], label="Val-Loss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid(alpha=0.3)
plt.show()

# Distribution of probabilities

plt.hist(probs_test.detach().cpu().numpy(), bins=50)

# ROC-AUC score

roc_auc = roc_auc_score(y_test.long().cpu().numpy(), probs_test.cpu().numpy())
print("ROC-AUC:", roc_auc)

# F1 score

f1 = f1_score(y_test.cpu(), preds.cpu())
print("F1:", f1)

# Balanced accuracy score

bal_acc = balanced_accuracy_score(y_test.cpu().numpy(), preds.cpu().numpy())
print("Balanced acc:", bal_acc)

# Baseline: always predict majority class

majority_class = int(y_tr.mean() >= 0.5)
majority_preds = torch.full_like(y_test.long(), majority_class)
majority_acc = (majority_preds == y_test.long()).float().mean().item()

print(f"Majority class accuracy: {majority_acc*100:.2f}%")