import itertools
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from classifier.data_processing import trim_set
from classifier.train_helpers import train_one_fold
from classifier.model import make_model

def run_time_series_cv(hp, X, y, idxs, n_splits, device, horizon, lookback, **train_kwargs):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(np.arange(len(y))), 1):

        tr_idx = trim_set(tr_idx, val_idx, idxs, horizon)

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = make_model(hp, lookback, train_kwargs["n_features"])

        # Normalize data for easier learning

        mean = X_tr.mean(dim=(0,1), keepdims=True)
        std  = X_tr.std(dim=(0,1), keepdims=True) + 1e-8

        X_tr = (X_tr - mean) / std
        X_val = (X_val - mean) / std

        metrics = train_one_fold(
            model, X_tr, y_tr, X_val, y_val,
            device=device, **train_kwargs
        )
        metrics["fold"] = fold
        metrics["train_n"] = len(tr_idx)
        metrics["val_n"] = len(val_idx)
        results.append(metrics)
        print(f"Fold {fold}: loss={metrics['val_loss']:.4f} acc={metrics['acc']:.3f} "
              f"f1={metrics['f1']:.3f} auc={metrics['roc_auc']:.3f}")

    # simple aggregate
    def avg(key):
        vals = [r[key] for r in results if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nCV mean:",
          f"loss={avg('val_loss'):.4f}",
          f"acc={avg('acc'):.3f}",
          f"f1={avg('f1'):.3f}",
          f"auc={avg('roc_auc'):.3f}")

    return results

def tune_hparams_ts_cv(
        X, y, idxs, horizon, lookback,
        param_grid=None,
        n_trials=20,
        metric="roc_auc",
        n_splits=4,
        device="cpu",
        train_kwargs=None,
    ):


    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[param_grid[k] for k in keys])]

    best_score = -np.inf
    best_hp = None
    best_results = None

    for i, hp in enumerate(combos, 1):
        # model = make_model(hp)
        print(f"\n--- Trial {i}/{len(combos)} | hp={hp} ---")
        results = run_time_series_cv(
            hp, X, y, idxs,
            n_splits=n_splits,
            device=device,
            horizon=horizon,
            lookback=lookback,
            **train_kwargs,
        )

        # average metric across folds
        vals = [r[metric] for r in results if not np.isnan(r[metric])]
        score = float(np.mean(vals)) if vals else (float("nan"))

        print(f"Trial score ({metric} mean): {score}")

        if np.isnan(score):
            continue

        if score > best_score:
            best_score = score
            best_hp = hp
            best_results = results
            print("--- NEW BEST ---")

    print("\nBEST:")
    print("hp:", best_hp)
    print("score:", best_score)
    return best_hp, best_score, best_results

