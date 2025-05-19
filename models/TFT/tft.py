import math
import sys
import warnings
from pathlib import Path

import numpy

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import r2_score

CWD = Path(__file__).parent.resolve()
ROOT = CWD.parent.parent.resolve()
sys.path.append(str(ROOT / "models"))


from preprocess import VN30, preprocess_v2  # type: ignore  # noqa


warnings.filterwarnings("ignore")


class TFT(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,    # số feature mỗi bước thời gian
        d_model: int = 32,     # chiều embedding
        num_heads: int = 4,    # số head
        d_ff: int = 64,        # chiều hidden của feed-forward
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        # 1) Embedding không thay đổi thứ tự time: (batch, seq_len, input_dim) → (batch, seq_len, d_model)
        self.embed = nn.Linear(input_dim, d_model)

        # 2) Linear cho Q, K, V mỗi head gộp chung
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        # 3) FFN nhỏ để map context → delta_price
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, input_dim)
        return: (batch, input_dim)  -- dự đoán giá của bước tiếp theo
        """
        B, T, _ = x.shape

        # --- 1) Embed ---
        x_e = self.embed(x)                # (B, T, d_model)

        # --- 2) Query = last step, Key/Value = toàn chuỗi ---
        q = self.Wq(x_e[:, -1:, :])        # (B, 1, d_model)
        k = self.Wk(x_e)                   # (B, T, d_model)
        v = self.Wv(x_e)                   # (B, T, d_model)

        # scaled dot-product attention
        # (B, 1, d_model) @ (B, d_model, T) → (B, 1, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model)
        weights = F.softmax(scores, dim=-1)   # (B, 1, T)
        weights = self.attn_drop(weights)

        # context vector
        # (B, 1, T) @ (B, T, d_model) → (B, 1, d_model)
        context = weights @ v                # (B, 1, d_model)
        context = context.squeeze(1)         # (B, d_model)

        # --- 3) Skip connection: lấy embedding của bước cuối cùng ---
        last_embed = x_e[:, -1, :]           # (B, d_model)

        # --- 4) Ghép context + last_embed rồi qua FFN để predict delta ---
        rep = torch.cat([context, last_embed], dim=-1)  # (B, 2*d_model)
        delta = self.ffn(rep)                            # (B, input_dim)

        # --- 5) Dự đoán = giá cuối cùng + delta ---
        # nhớ x là giá đã được scaler transform → delta cũng phù hợp scale
        y_pred = x[:, -1, :] + delta                   # (B, input_dim)
        return y_pred


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = TFT()
# with open(CWD / "tft.csv", "w", encoding="utf-8", buffering=1) as csv:
#     csv.write("Checkpoints/Tests,")
#     csv.write(",".join(VN30))
#     csv.write("\n")

#     with torch.no_grad():
#         for i, ckpt_symbol in enumerate(VN30):
#             csv.write(ckpt_symbol)
#             model.load_state_dict(torch.load(CWD / "checkpoints" / f"tft_{ckpt_symbol}.pth", map_location=device))
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
