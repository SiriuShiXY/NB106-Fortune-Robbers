import torch

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, dates, codes):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).long()
        self.dates, self.codes = dates, codes

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y[idx],
            str(self.dates[idx]),
            str(self.codes[idx])
        )