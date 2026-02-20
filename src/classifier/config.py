import torch

HORIZON = 5 # approx 3 weeks
LOOKBACK = 10 # approx 5 weeks
STRIDER = 5
TEST_SIZE = 0.2
N_FEATURES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_KWARGS = {
    "n_features": N_FEATURES,
    "epochs":60,
    "batch_size":32,
    "lr":1e-4,
    "weight_decay":1e-2,
    "clip_grad":1.0,
    "patience":10
}