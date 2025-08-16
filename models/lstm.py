import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,      # SỬA: Tổng số feature sau khi preprocess (ví dụ: 120)
        hidden_dim: int = 64,  # Có thể tăng hidden_dim
        num_classes: int = 5,  # SỬA: Số lớp đầu ra
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        # 1) Lớp LSTM: input_size bây giờ là tổng số feature
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # 2) Lớp Linear cuối cùng: Map từ hidden state ra số lớp
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, input_dim) -- Dữ liệu 2D từ preprocess
        return: (batch, num_classes) -- Logits cho mỗi lớp
        """
        # --- SỬA Ở ĐÂY: Thêm một chiều để tạo chuỗi có độ dài 1 ---
        # (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)

        # --- 1) Đưa chuỗi (dài 1) vào LSTM ---
        # lstm_out shape: (batch, 1, hidden_dim)
        lstm_out, _ = self.lstm(x)

        # --- 2) Chỉ lấy output của bước thời gian cuối cùng (cũng là duy nhất) ---
        # (batch, 1, hidden_dim) -> (batch, hidden_dim)
        last_time_step_out = lstm_out[:, -1, :]

        # --- 3) Đưa qua lớp fully-connected để có dự đoán cuối cùng ---
        # (batch, hidden_dim) -> (batch, num_classes)
        logits = self.fc(last_time_step_out)
        
        return logits