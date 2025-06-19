import torch, torch.nn as nn
from config import Hyper

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=Hyper.HIDDEN_SIZE,
                 num_layers=Hyper.NUM_LAYERS, fc1=100, fc2=30):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=0.2)
        self.net  = nn.Sequential(
            nn.Linear(hidden_size, fc1), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fc1, fc2),         nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(fc2, 1),           nn.Sigmoid(),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.net(h[-1]).squeeze(-1)