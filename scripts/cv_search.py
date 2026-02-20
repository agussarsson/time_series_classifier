import json
import classifier.data_processing
from classifier.config import HORIZON, DEVICE
from classifier.cv_tuning import tune_hyperparams_cv
from classifier.data_processing import run_processing
from classifier.config import HORIZON, LOOKBACK, DEVICE, STRIDER, TEST_SIZE, N_FEATURES, TRAIN_KWARGS
from classifier.cv_tuning import tune_hparams_ts_cv
from sklearn.model_selection import TimeSeriesSplit
from classifier.data_processing import trim_set



X_trval, y_trval, X_test, y_test, idxs_trval = run_processing("AAPL", LOOKBACK, HORIZON, STRIDER, TEST_SIZE)

param_grid = {
    "d_model": [16, 32, 64],
    "nhead":  [2, 4, 8],
    "nlayers":[1, 2],
    "dropout":[0.1, 0.2],
}

best_hp, best_score, _ = tune_hparams_ts_cv(
    X_trval, y_trval, idxs_trval, HORIZON, LOOKBACK,
    param_grid=param_grid,
    metric="roc_auc",
    n_splits=4,
    device=DEVICE,
    train_kwargs=TRAIN_KWARGS,
)

with open("./search_results/cv_search_results.json", "w") as f:
    json.dump(best_hp, f)


print("\nTraining final model with best hyperparameters:", best_hp)

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

with open("./search_results/cv_search_results.json", "w") as f:
    json.dump(best_hp, f)