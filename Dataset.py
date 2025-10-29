from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data, T, device='cuda'):
        super().__init__()
        self.data = data
        self.T = T # 10
        self.device = device
        self.max_len = data.shape[1] - T

    def __len__(self):
        return max(0, self.max_len)

    def __getitem__(self, idx):
        X = self.data[:, idx:idx+self.T, :]
        label = (self.data[:, idx+self.T, 0] - X[:, -1, 0]) > 0
        label = label.long()
        return X, label