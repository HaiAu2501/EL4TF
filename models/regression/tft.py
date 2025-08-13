import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalFusionTransformer(nn.Module):
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