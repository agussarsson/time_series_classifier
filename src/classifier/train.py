import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score
import torch.nn as nn

def train_one_fold(model, X_tr, y_tr, X_val, y_val,
            device="cpu", **train_kwargs):

    val_losses = []
    train_losses = []
    loss_counter = 0
    prev_loss = np.inf
    best_val = np.inf
    best_state = None

    lr = train_kwargs["lr"]
    weight_decay = train_kwargs["weight_decay"]
    batch_size = train_kwargs["batch_size"]
    epochs = train_kwargs["epochs"]
    clip_grad = train_kwargs["clip_grad"]
    patience = train_kwargs["patience"]

    model.to(device)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
        )
    loss_fn = nn.BCEWithLogitsLoss()


    for i in range(epochs):
        # # Random batch suffling
        perm = torch.randperm(X_tr.shape[0], device=device)
        X_tr = X_tr[perm]
        y_tr = y_tr[perm]
        model.train()

        train_loss = 0

        num_batches = 0

        for start in range(0, X_tr.shape[0], batch_size):
            end = start + batch_size

            Xbatch = X_tr[start : end, :, :]
            ybatch = y_tr[start : end]

            logits = model(Xbatch)
            loss = loss_fn(logits, ybatch)

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()

            train_loss += loss.item()
            num_batches += 1

        # Evaluate performance and check early stopping
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            probs = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_loss = loss_fn(val_logits, y_val)

            y_true = y_val.detach().cpu().numpy().astype(int)


        preds = (probs > 0.5).astype(int)

        if val_loss.item() < best_val - 1e-4:
            best_val = val_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            loss_counter = 0
        else:
            loss_counter += 1

        if loss_counter >= patience:
            break

        prev_loss = val_loss.item()
        val_losses.append(val_loss.item())
        train_losses.append(train_loss / num_batches)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        probs = torch.sigmoid(val_logits).cpu().numpy()

    preds = (probs > 0.5).astype(int)

    out = {
        "val_loss": best_val,
        "acc": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_state": best_state
    }
    return out