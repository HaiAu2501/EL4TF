import torch.nn as nn

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, n_channels=4, hidden_dim=64, output_dim=4, p_dropout=0.2):
        super().__init__()
        # Conv block với BatchNorm + Dropout
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p_dropout),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)  # gom về (batch, 64, 1)

        # MLP head với thêm dropout
        self.fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch, n_channels, window_size)
        x = self.conv_block(x)       # → (batch, 64, window_size)
        x = self.pool(x).squeeze(-1) # → (batch, 64)
        x = self.fc(x)               # → (batch, output_dim)
        return x