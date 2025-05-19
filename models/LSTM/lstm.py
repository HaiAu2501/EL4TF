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


class LSTM(nn.Module):
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


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LSTM(
#     n_features=4,    # [open, high, low, close]
#     n_layers=1,
#     hidden_dim=64,
#     fc_dim=32,
#     output_dim=4,    # dự báo [open, high, low, close]
#     dropout=0.3
# )
# with open(CWD / "lstm.csv", "w", encoding="utf-8", buffering=1) as csv:
#     csv.write("Checkpoints/Tests,")
#     csv.write(",".join(VN30))
#     csv.write("\n")

#     with torch.no_grad():
#         for i, ckpt_symbol in enumerate(VN30):
#             csv.write(ckpt_symbol)
#             model.load_state_dict(torch.load(CWD / "checkpoints" / f"lstm_{ckpt_symbol}.pth", map_location=device))
#             model.eval()

#             for j, test_symbol in enumerate(VN30):
#                 _, _, test_loader, scaler = preprocess_v2(test_symbol, "rnn")

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
