import sys
import warnings
from pathlib import Path

import numpy

import torch
from torch import nn
from sklearn.metrics import r2_score

CWD = Path(__file__).parent.resolve()
ROOT = CWD.parent.parent.resolve()
sys.path.append(str(ROOT / "models"))


from preprocess import VN30, preprocess_v2  # type: ignore  # noqa


warnings.filterwarnings("ignore")


class TCN(nn.Module):
    def __init__(self, n_channels=4, window_size=30, hidden_dim=256, output_dim=4, p_dropout=0.1):
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
        x = self.pool(x).squeeze(-1)  # → (batch, 64)
        x = self.fc(x)               # → (batch, output_dim)
        return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = TCN(
#     n_channels=4,
#     window_size=30,
#     hidden_dim=256,
#     output_dim=4,
#     p_dropout=0.1
# ).to(device)


# with open(CWD / "tcn.csv", "w", encoding="utf-8", buffering=1) as csv:
#     csv.write("Checkpoints/Tests,")
#     csv.write(",".join(VN30))
#     csv.write("\n")

#     with torch.no_grad():
#         for i, ckpt_symbol in enumerate(VN30):
#             csv.write(ckpt_symbol)
#             model.load_state_dict(torch.load(CWD / "checkpoints" / f"tcn_{ckpt_symbol}.pth", map_location=device))
#             model.eval()

#             for j, test_symbol in enumerate(VN30):
#                 _, _, test_loader, scaler = preprocess_v2(test_symbol, "cnn")

#                 all_preds = []
#                 all_targets = []
#                 for X_batch, y_batch in test_loader:
#                     X_batch = X_batch.to(device)
#                     preds = model(X_batch).cpu().numpy()
#                     all_preds.append(preds)
#                     all_targets.append(y_batch.numpy())

#                 all_preds = numpy.vstack(all_preds)   # (n_samples, 5)
#                 all_targets = numpy.vstack(all_targets)

#                 # Inverse scaling
#                 all_preds_inv = scaler.inverse_transform(all_preds)
#                 all_targets_inv = scaler.inverse_transform(all_targets)

#                 # Tính metrics
#                 r2 = r2_score(all_targets_inv, all_preds_inv, multioutput='uniform_average')
#                 csv.write(f",{r2}")

#             csv.write("\n")
