# Trend Transformer — Time Series Classification

Transformer-based deep learning model for **financial time-series trend prediction** using leakage-safe evaluation and walk-forward validation.

---

## Overview

The goal is to predict whether the **short-term price trend** of a financial asset will be positive or negative over a future horizon based on historical data.

Given a rolling window of historical features, the model learns to classify:

1 if the future trend is > 1
0 if the future trend is <= 0

Where the future trend is decided by the regression slope over the log-prices in the prediction horizon.

---

## Features:

Derived from historical data from yfinance.

- log returns  
- rolling realized volatility  
- normalized trading volume (z-score)

Features are derived using rolling statistics to maintain temporal causality.

---

## Leakage prevention

To avoid leakage, the train, validation-and test-sets are carefully crafted so that labels (derived from horizons) are contained within each set.

---

## Installation

Create a virtual environment and install the project:

```bash
git clone https://github.com/<username>/time_series_classifier.git
cd time_series_classifier

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .

```

---

## Running the program

Hyperparameter search:

```bash
python scripts/cv_search.py
```

The best hyperparameters will be saved in ```search_results/cv_search_results.json```.

Training and evaluating the best model:

```bash
python scripts/train_classifier.py
```

The best model will be saved in ```models/final_model.pt```.