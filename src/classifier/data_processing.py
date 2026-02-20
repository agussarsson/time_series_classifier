import torch
import numpy as np
import yfinance as yf
from scipy import stats


def labels_from_series_feats(feats, close_log, horizon, lookback, stride):
    """
    feats: (T, F) torch.float32
    close_log: (T,) np array aligned with feats index (after dropna)
    returns: X (M, lookback, F), y (M,) binary
    """
    T = feats.shape[0]
    idxs = np.arange(lookback, T - horizon + 1, stride)
    M = len(idxs)

    X = torch.empty((M, lookback, feats.shape[1]), dtype=torch.float32)
    y = torch.empty((M,), dtype=torch.float32)

    t = np.arange(horizon)

    for k, i in enumerate(idxs):
        X[k] = feats[i - lookback : i]  # past window

        # label: sign of future slope on *log price* (more stable than raw price)
        future_logp = close_log[i : i + horizon]
        slope, *_ = stats.linregress(t, future_logp)
        y[k] = 1.0 if slope > 0 else 0.0

    return X, y, idxs


def trim_set(left_idx, right_idx, idxs, horizon):
    boundary_t = idxs[right_idx[0]]
    label_end_t = idxs[left_idx] + (horizon - 1)
    return left_idx[label_end_t < boundary_t]


def run_processing(ticker, lookback, horizon, strider, test_size):

    stock_history_data = yf.Ticker(str(ticker)).history(period="4y")
    stock_history_data.head()

    price_history = torch.from_numpy(stock_history_data["Close"].to_numpy()).unsqueeze(dim=1)
    volume_history = torch.from_numpy(stock_history_data["Volume"].to_numpy()).unsqueeze(dim=1)
    # print(len(price_history), len(volume_history))
    historical_data = torch.cat([price_history, volume_history], dim=1)
    print(historical_data.shape)

    df = stock_history_data.copy()

    df["log_close"] = np.log(df["Close"])
    df["log_ret"] = df["log_close"].diff()
    df["abs_ret"] = df["log_ret"].abs()

    df["rv"] = df["log_ret"].rolling(lookback, min_periods=lookback).std()

    vol_mean = df["Volume"].rolling(lookback, min_periods=lookback).mean()
    vol_std  = df["Volume"].rolling(lookback, min_periods=lookback).std().replace(0, np.nan)
    df["vol_z"] = (df["Volume"] - vol_mean) / vol_std

    df = df.dropna(subset=["log_ret", "rv", "vol_z"]).copy()

    feat_cols = ["log_ret", "rv", "vol_z"]
    features = torch.tensor(df[feat_cols].to_numpy(), dtype=torch.float32)  # (T, 3)

    close_log = df["log_close"].to_numpy()
    X, y, idxs = labels_from_series_feats(features, close_log, horizon, lookback, strider)


    # Save a test set before train/val split

    n = len(y)

    test_len = int(test_size * len(y))
    test_idx = np.arange(n-test_len, n)

    split_point = n - test_len

    boundary_t = idxs[split_point]
    trval_mask = (idxs + (horizon - 1)) < boundary_t

    X_trval = X[trval_mask]
    y_trval = y[trval_mask]

    idxs_trval = idxs[trval_mask]

    X_test, y_test = torch.tensor(X[test_idx,:]).float(), torch.tensor(y[test_idx]).float()

    return X_trval, y_trval, X_test, y_test, idxs_trval

