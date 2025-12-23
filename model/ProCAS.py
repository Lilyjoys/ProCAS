import torch
import torch.nn as nn
import torch.nn.functional as F


class ProCAS(nn.Module):

    def __init__(self, input_dim=430,
                 class_num=1,
                 fc_hidden=128,    #  FC
                 se_ratio=8,
                 lstm_hidden=16,   #  16
                 attn_heads=1):
        super(ProCAS, self).__init__()

        self.fc1 = nn.Linear(input_dim, fc_hidden)
        self.bn1 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.bn2 = nn.BatchNorm1d(fc_hidden)

        # —— Self-Highway ——
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

        #  LSTM
        x_seq = x.unsqueeze(1)              # (B,1,fc_hidden)
        lstm_out, _ = self.lstm(x_seq)      # (B,1,2*lstm_hidden)
        # Self-Attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        x2 = self.ln(lstm_out + attn_out)   # (B,1,2*h)
        pooled = x2.squeeze(1)              # (B,2*h)

        logits = self.classifier(pooled).squeeze(-1)
        return torch.sigmoid(logits)