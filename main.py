import torch
from dataProcess import process_data
import os
import numpy as np
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from Lysine.Acetyl.AceMain import Acetyl_train
from Lysine.Crotonyl.Crotonyl import Crotonyl_train
from Lysine.Methyl.Methyl import Methyl_train
from Lysine.Succinyl.Succinyl import Succinyl_train
from train import Mult_train
import torch
import torch.nn as nn
import torch.nn.functional as F
from Metrics import multi_label_metrics_binary, compute_binary_metrics


class ProCAS(nn.Module):

    def __init__(self, input_dim=430,
                 class_num=1,
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

        self.attn = nn.MultiheadAttention(embed_dim=2*lstm_hidden, num_heads=attn_heads, batch_first=True)
        self.ln = nn.LayerNorm(2*lstm_hidden)


        self.classifier = nn.Sequential(
            nn.Linear(2*lstm_hidden, 16),  # 16
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, class_num)
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


        x_seq = x.unsqueeze(1)              # (B,1,fc_hidden)
        lstm_out, _ = self.lstm(x_seq)      # (B,1,2*lstm_hidden)
        # Self-Attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        x2 = self.ln(lstm_out + attn_out)   # (B,1,2*h)
        pooled = x2.squeeze(1)              # (B,2*h)


        logits = self.classifier(pooled).squeeze(-1)
        return torch.sigmoid(logits)



def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_randomseed(40)
    import pickle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = "./dataset/data_cache.pkl"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            X_train, X_train2, Y_train, label_train, X_test, X_test2, Y_test, label_test, ctd_train, ctd_test = pickle.load(f)
    else:
        X_train, X_train2, Y_train, label_train, X_test, X_test2, Y_test, label_test, ctd_train, ctd_test = process_data(
            "./dataset/Train dataset", "./dataset/Test dataset")
        with open(cache_path, "wb") as f:
            pickle.dump((X_train, X_train2, Y_train, label_train, X_test, X_test2, Y_test, label_test, ctd_train, ctd_test), f)

    label1_train, label2_train, label3_train, label4_train = [label_train[:, i] for i in range(4)]
    label1_test, label2_test, label3_test, label4_test = [label_test[:, i] for i in range(4)]

    # Acetyl_train(X_train, label1_train, X_test, label1_test, X_train2, X_test2, ctd_train, ctd_test)
    # Crotonyl_train(X_train, label2_train, X_test, label2_test, X_train2, X_test2, ctd_train, ctd_test)
    # Methyl_train(X_train, label3_train, X_test, label3_test, X_train2, X_test2, ctd_train, ctd_test)
    # Succinyl_train(X_train, label4_train, X_test, label4_test, X_train2, X_test2, ctd_train, ctd_test)
    # Mult_train(X_train, X_train2, Y_train, label_train, X_test, X_test2, Y_test, label_test, ctd_train, ctd_test)

    X_train = np.concatenate((X_train2 * 1e-5, X_train * 5, ctd_train * 1e-5), axis=1)
    X_test = np.concatenate((X_test2 * 1e-5, X_test * 5, ctd_test * 1e-5), axis=1)


    input_dim = X_test.shape[1]
    model1 = ProCAS(input_dim=X_train.shape[1])
    model1.load_state_dict(torch.load("./Lysine/best_ckpt/Acetyl_binary.pth"))
    model2 = ProCAS(input_dim=X_train.shape[1])
    model2.load_state_dict(torch.load("./Lysine/best_ckpt/Crotonyl_binary.pth"))
    model3 = ProCAS(input_dim=X_train.shape[1])
    model3.load_state_dict(torch.load("./Lysine/best_ckpt/Methyl_binary.pth"))
    model4 = ProCAS(input_dim=X_train.shape[1])
    model4.load_state_dict(torch.load("./Lysine/best_ckpt/Succinyl_binary.pth"))

    model_multi = ProCAS(input_dim=X_train.shape[1], class_num=4)
    model_multi.load_state_dict(torch.load("./Lysine/best_ckpt/Multi.pth"))

    X_test = torch.tensor(X_test, dtype=torch.float32)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    proba_0 = model1(X_test)
    proba_1 = model2(X_test)
    proba_2 = model3(X_test)
    proba_3 = model4(X_test)

    compute_binary_metrics(label3_test, proba_2.detach().numpy(), proba_2.detach().numpy())

    #  (n_samples, 4)
    pros_1 = np.column_stack(
        (proba_0.detach().numpy(), proba_1.detach().numpy(), proba_2.detach().numpy(), proba_3.detach().numpy()))
    model_multi.eval()
    pros_2 = model_multi(X_test).detach().numpy()

    print(pros_2)

    print(multi_label_metrics_binary(label_test, pros_1))
    print(multi_label_metrics_binary(label_test, pros_2))

    print(multi_label_metrics_binary(label_test, pros_1 * 0.4 + pros_2 * 0.6))