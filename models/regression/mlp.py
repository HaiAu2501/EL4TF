# mlp.py
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def bits_to_upper_tri(mask_bits: List[int], L: int) -> torch.Tensor:
    """
    Chuyển danh sách bit (length = L*(L-1)/2) thành ma trận skip (L x L),
    chỉ cho phép j < i. 1 nghĩa là có skip từ layer j -> layer i.
    """
    expected = L * (L - 1) // 2
    if len(mask_bits) < expected:
        # pad 0 nếu thiếu
        mask_bits = list(mask_bits) + [0] * (expected - len(mask_bits))
    mask_bits = mask_bits[:expected]

    idx = 0
    M = torch.zeros((L, L), dtype=torch.uint8)
    for i in range(1, L):
        for j in range(0, i):
            M[i, j] = 1 if mask_bits[idx] else 0
            idx += 1
    return M  # shape [L, L], upper-tri (strictly below diagonal)

class MultiLayerPerception(nn.Module):
    """
    MLP nhỏ gọn cho regression, genome chỉ mã hoá skip connections giữa các hidden layers.
    - Số layer ẩn và hidden_dim cố định nhỏ để train nhanh.
    - Tất cả hidden layers có cùng hidden_dim -> có thể cộng skip trực tiếp (không cần adapter).
    - Base connection: i-1 -> i (Linear).
    - Skip connections: j -> i với j < i (cộng vào trước khi kích hoạt).

    Kiến trúc:
      in_dim -> [Linear -> Act -> Dropout] x 1 (tạo h1)
      Với i=2..L: out = Linear(h_{i-1}) + sum_{j<i-1, skip[i,j]=1} h_j; rồi Act + Dropout
      head: Linear(hidden_dim -> out_dim)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 4,
        skip_bits: List[int] = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: số chiều đặc trưng đầu vào
            output_dim: số đầu ra (vd 4)
            hidden_dim: kích thước mỗi hidden layer (cố định)
            num_layers: số hidden layers (L)
            skip_bits: danh sách bit chiều dài L*(L-1)/2 (chỉ 0/1),
                       bit cho j<i để bật skip j -> i. Nếu None thì không có skip.
            dropout: xác suất dropout sau mỗi hidden
            activation: "relu" | "gelu" | "tanh"
        """
        super().__init__()
        assert num_layers >= 1, "num_layers phải >= 1"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Kích hoạt
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Base linears: in->h1, và h(i-1)->hi cho i=2..L
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))  # tạo h1
        for _ in range(1, num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))  # h(i-1)->hi
        self.base_linears = nn.ModuleList(layers)

        # Skip matrix (L x L) với dtype uint8
        if skip_bits is None:
            skip_bits = [0] * (num_layers * (num_layers - 1) // 2)
        skip_mat = bits_to_upper_tri(skip_bits, num_layers)  # [L, L]
        self.register_buffer("skip_mat", skip_mat, persistent=True)

        # Head
        self.head = nn.Linear(hidden_dim, output_dim)

        # Optional: layernorm nhẹ để ổn định (không bắt buộc)
        self.norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        return: [B, output_dim]
        """
        h_list = []

        # h1
        h = self.base_linears[0](x)
        h = self.norms[0](h)
        h = self.act(h)
        h = self.drop(h)
        h_list.append(h)  # index 0

        # h2..hL
        for i in range(1, self.num_layers):
            out = self.base_linears[i](h_list[i - 1])  # base
            # add skips j -> i
            row = self.skip_mat[i]  # [L], skip_mat[i, j] with j<i valid
            if row.any():
                # chỉ cộng các h_j với j<i có bit=1
                for j in range(0, i):
                    if row[j].item():
                        out = out + h_list[j]
            out = self.norms[i](out)
            out = self.act(out)
            out = self.drop(out)
            h_list.append(out)

        y = self.head(h_list[-1])
        return y  # [B, output_dim]
