import torch.nn as nn

class LongShortTermMemory(nn.Module):
    def __init__(self, n_features=4, n_layers=1, hidden_dim=64, fc_dim=32, output_dim=4, dropout=0.3):
        super().__init__()
        # LSTM với dropout giữa các layer (chỉ active khi n_layers > 1)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        # Dropout sau khi lấy last-step output
        self.dropout = nn.Dropout(dropout)
        # FC phụ: hidden_dim → fc_dim → output_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, n_features)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_dim)
        last = lstm_out[:, -1, :]
        # last: (batch_size, hidden_dim)
        dropped = self.dropout(last)
        # dropped: (batch_size, hidden_dim)
        y_pred = self.fc(dropped)
        # y_pred: (batch_size, output_dim)
        return y_pred