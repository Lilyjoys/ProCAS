from Metrics import compute_binary_metrics
from Metrics import multi_label_metrics_binary
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import os
from Resample.MiDpp import support_points_undersample
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import copy

class ProCAS(nn.Module):

    def __init__(self, input_dim=430,
                 fc_hidden=128,
                 se_ratio=8,
                 lstm_hidden=16,
                 attn_heads=1):
        super(ProCAS, self).__init__()

        self.fc1 = nn.Linear(input_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.bn2 = nn.BatchNorm1d(fc_hidden)

        self.highway_H = nn.Linear(fc_hidden, fc_hidden)
        self.highway_T = nn.Linear(fc_hidden, fc_hidden)

        # Squeeze-and-Excitation
        self.se_fc1 = nn.Linear(fc_hidden, fc_hidden // se_ratio)
        self.se_fc2 = nn.Linear(fc_hidden // se_ratio, fc_hidden)

        # BiLSTM
        self.lstm = nn.LSTM(fc_hidden, lstm_hidden, batch_first=True, bidirectional=True)
        #
        self.attn = nn.MultiheadAttention(embed_dim=2*lstm_hidden, num_heads=attn_heads, batch_first=True)
        self.ln = nn.LayerNorm(2*lstm_hidden)


        self.classifier = nn.Sequential(
            nn.Linear(2*lstm_hidden, 16),  # 保持 16
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        # x: (B,430)
        x = F.relu(self.bn1(self.fc1(x)))   # (B, fc_hidden)
        x = F.relu(self.bn2(self.fc2(x)))   # (B, fc_hidden)

        # Self-Highway: H,T gating
        H = F.relu(self.highway_H(x))       # (B, fc_hidden)
        T = torch.sigmoid(self.highway_T(x))# (B, fc_hidden)
        x = H * T + x * (1 - T)             # gated combination

        # SE
        w = x.mean(dim=1)                   # (B,)
        w = F.relu(self.se_fc1(x))
        w = torch.sigmoid(self.se_fc2(w))   # (B, fc_hidden)
        x = x * w                           # (B, fc_hidden)

        #  LSTM
        x_seq = x.unsqueeze(1)              # (B,1,fc_hidden)
        lstm_out, _ = self.lstm(x_seq)      # (B,1,2*lstm_hidden)
        # Self-Attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        x2 = self.ln(lstm_out + attn_out)   # (B,1,2*h)
        pooled = x2.squeeze(1)              # (B,2*h)


        logits = self.classifier(pooled).squeeze(-1)
        return torch.sigmoid(logits)



def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)\.pth', filename)
    return int(match.group(1)) if match else -1


def Mult_train(X_train, X_train2, Y_train, label_train, X_test, X_test2, Y_test, label_test, ctd_train, ctd_test, isResample=0,
                 save_dir="./checkpoints", batch_size=256, epochs=700, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isResample == 1:
        re_X_train, re_Y_train, index = support_points_undersample(X_train, Y_train, 1, 1000)
        X_train, label_train = X_train[index], label_train[index]
        X_train2 = X_train2[index]
        ctd_train = ctd_train[index]
        label_counts = Counter(re_Y_train)
        print(label_counts)


    X_train = np.concatenate((X_train2*1e-5, X_train*5, ctd_train*1e-5), axis=1)
    X_test = np.concatenate((X_test2*1e-5, X_test*5, ctd_test*1e-5), axis=1)

    os.makedirs(save_dir, exist_ok=True)

    model = ProCAS(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_tr = torch.tensor(label_train, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_te = torch.tensor(label_test, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_tr, Y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    fold_metrics_list, avg_metrics = cross_validate_model(X_train, label_train,model_class=ProCAS,input_dim=X_train.shape[1],device=device,batch_size=batch_size,epochs=epochs,lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:03d} - Loss: {avg_loss:.4f}")


        if epoch % 5 == 0:
            ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved model at: {ckpt_path}")


    print("\n===== Testing all saved models =====")
    checkpoint_files = sorted(
        [f for f in os.listdir(save_dir) if f.endswith('.pth')],
        key=extract_epoch
    )

    for ckpt in checkpoint_files:
        ckpt_path = os.path.join(save_dir, ckpt)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        with torch.no_grad():
            y_prob = model(X_te)
            y_pred = (y_prob >= 0.5).float()

        metrics = multi_label_metrics_binary(Y_te.cpu().numpy(), y_pred.cpu().numpy())
        print(f"\n=== Evaluation for {ckpt} ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


def cross_validate_model(X, Y, model_class, input_dim, device='cuda',
                         batch_size=128, epochs=300, lr=1e-4, verbose=True):

    import copy
    from sklearn.model_selection import KFold
    from torch.utils.data import DataLoader, TensorDataset

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    kf = KFold(n_splits=5, shuffle=True)
    fold_metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tensor)):
        print(f"\n===== Fold {fold + 1} =====")

        model = model_class(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)
        criterion = nn.BCELoss()

        train_loader = DataLoader(TensorDataset(X_tensor[train_idx], Y_tensor[train_idx]),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_tensor[val_idx], Y_tensor[val_idx]),
                                batch_size=batch_size)

        best_acc = 0.0
        best_metrics = None

        for epoch in range(epochs):
            # —— Train Phase ——
            model.train()
            total_loss = 0.0
            all_train_preds, all_train_trues = [], []

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

                all_train_preds.append(out.detach().cpu().numpy())
                all_train_trues.append(yb.cpu().numpy())

            avg_loss = total_loss / len(train_loader.dataset)

            # —— Evaluation Phase ——
            model.eval()
            with torch.no_grad():
                all_val_preds, all_val_trues = [], []
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    preds = model(xb).cpu().numpy()
                    all_val_preds.append(preds)
                    all_val_trues.append(yb.numpy())


            train_metrics = multi_label_metrics_binary(
                np.vstack(all_train_trues), np.vstack(all_train_preds)
            )
            val_metrics = multi_label_metrics_binary(
                np.vstack(all_val_trues), np.vstack(all_val_preds)
            )

            if val_metrics["Accuracy"] > best_acc:
                best_acc = val_metrics["Accuracy"]
                best_metrics = copy.deepcopy(val_metrics)


            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
                print(f"  ▶ Train Accuracy = {train_metrics['Accuracy']:.4f} | "
                      f"Coverage = {train_metrics['Coverage']:.4f} | "
                      f"Aiming = {train_metrics['Aiming']:.4f}")
                print(f"  ▶ Val   Accuracy = {val_metrics['Accuracy']:.4f} | "
                      f"Coverage = {val_metrics['Coverage']:.4f} | "
                      f"Aiming = {val_metrics['Aiming']:.4f}")

        print(f"✔️ Fold {fold + 1} Best Val Accuracy: {best_acc:.4f}")
        print(f"Best Metrics: {best_metrics}")
        fold_metrics_list.append(best_metrics)


    avg_metrics = {}
    for key in fold_metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in fold_metrics_list])
    print("\n===== 🧮 5-Fold Average Metrics =====")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    return fold_metrics_list, avg_metrics