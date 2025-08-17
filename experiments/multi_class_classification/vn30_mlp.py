import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

# 1. Thiết lập tham số
class Config:
    batch_size = 32
    epochs = 8
    seq_length = 5
    learning_rate = 0.001
    hidden_dim = 24
    num_workers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Kiến trúc mạng nơ-ron
class TinyPricePredictor(nn.Module):
    def __init__(self, input_dim=4, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * Config.seq_length, Config.hidden_dim),
            nn.ReLU(),
            nn.Linear(Config.hidden_dim, Config.hidden_dim // 2),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(Config.hidden_dim // 2, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        features = self.encoder(x)
        return self.classifier(features)

# 3. Xử lý dữ liệu
class StockDataset(Dataset):
    def __init__(self, dataframe, seq_length=5, scaler=None):
        self.data = dataframe.copy() # Sử dụng .copy() để tránh cảnh báo
        self.seq_length = seq_length
        self.data.columns = [col.lower() for col in self.data.columns]
        self.features = ['open', 'high', 'low', 'close']

        # Chuẩn hóa dữ liệu: fit trên train và transform cho cả train/test
        if scaler is None:
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(self.data[self.features])
        else:
            self.scaler = scaler
            scaled_values = self.scaler.transform(self.data[self.features])
        
        self.data[self.features] = scaled_values
        
        # Tạo nhãn (multi-class)
        self.create_labels()

    def create_labels(self):
        daily_returns = self.data['close'].pct_change().shift(-1)
        bins = [-np.inf, -0.02, 0, 0.02, np.inf]
        labels = pd.cut(daily_returns, bins=bins, labels=[0, 1, 2, 3])
        self.data['label'] = labels
        self.data = self.data.dropna()

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx:idx + self.seq_length][self.features].values
        label = self.data.iloc[idx + self.seq_length - 1]['label']
        return torch.FloatTensor(sequence), torch.LongTensor([label]).squeeze()

# 4. Hàm đánh giá
def quick_evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            outputs = model(X.to(Config.device))
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return balanced_accuracy_score(y_true, y_pred)

# 5. Huấn luyện 
def train_single_stock(model, train_path, test_path, stock_code):
    # Tải dữ liệu từ các tệp train và test
    train_df = pd.read_csv(train_path).sort_values('time')
    test_df = pd.read_csv(test_path).sort_values('time')
    
    # Tạo dataset và loader
    # Scaler sẽ được fit trên train_dataset
    train_dataset = StockDataset(train_df, Config.seq_length)
    # Scaler từ train_dataset được sử dụng lại cho test_dataset
    test_dataset = StockDataset(test_df, Config.seq_length, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=Config.num_workers)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=Config.batch_size,
                             shuffle=False)
    
    # Khởi tạo model
    model = model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Vòng lặp huấn luyện
    for epoch in range(Config.epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X.to(Config.device))
            loss = criterion(outputs, y.to(Config.device))
            loss.backward()
            optimizer.step()
            
        # Đánh giá sau mỗi 2 epoch
        if epoch % 2 == 0 or epoch == Config.epochs - 1:
            val_acc = quick_evaluate(model, test_loader)
            print(f'{stock_code} | Epoch {epoch}: Val Acc = {val_acc:.3f}')
            
    # Lưu model
    torch.save(model.state_dict(), f'checkpoints/tiny_{stock_code}.pth')
    final_acc = quick_evaluate(model, test_loader)
    return final_acc

# 6. Xử lý toàn bộ các tệp
def process_all_files(folder_path):
    results = []
    # Lấy danh sách các mã cổ phiếu duy nhất từ tên tệp train
    stock_codes = sorted(list(set(f.split('_')[0] for f in os.listdir(folder_path) if f.endswith('_train.csv'))))
    
    for stock_code in stock_codes:
        train_file = os.path.join(folder_path, f'{stock_code}_train.csv')
        test_file = os.path.join(folder_path, f'{stock_code}_test.csv')
        
        # Kiểm tra xem cả hai tệp train và test có tồn tại không
        if os.path.exists(train_file) and os.path.exists(test_file):
            model = TinyPricePredictor(num_classes=4)
            
            print(f'\nTraining on {stock_code}...')
            acc = train_single_stock(model, train_file, test_file, stock_code)
            
            results.append({
                'Stock': stock_code,
                'Balanced Accuracy': acc,
                'Model Path': f'checkpoints/tiny_{stock_code}.pth'
            })
        else:
            print(f"Bỏ qua {stock_code}: Thiếu tệp train hoặc test.")
            
    # Lưu kết quả tổng hợp
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiments/multi_class_classification/multi_class_results.csv', index=False)
    return results_df

# 7. Main execution (Đã cập nhật đường dẫn)
if __name__ == "__main__":
    # Tạo các thư mục cần thiết
    os.makedirs('checkpoints', exist_ok=True)
    # os.makedirs('experiments', exist_ok=True)
    
    # Đường dẫn đến thư mục chứa các tệp CSV của bạn
    folder_path = r'D:\Github_Local_path\EL4TF\data\vn30\multi_class_classification'
    
    final_results = process_all_files(folder_path)
    
    print("\n--- Final Results ---")
    print(final_results)