import torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from pathlib import Path
from config import Hyper, Paths, logger
from models.lstm import LSTMModel
from datasets.stock_dataset import StockDataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class LSTMTrainer:
    def __init__(self, X_train, y_train, X_val, y_val, fold=None):
        self.fold = fold
        self.train_loader = DataLoader(StockDataset(*X_train), Hyper.BATCH_SIZE, shuffle=True)
        self.val_loader   = DataLoader(StockDataset(*X_val),  Hyper.BATCH_SIZE)
        self.model  = LSTMModel(input_size=X_train[0].shape[-1]).to(DEVICE)
        self.optim  = torch.optim.Adam(self.model.parameters(), lr=Hyper.LR)
        self.crit   = torch.nn.BCELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, 'min', patience=5, factor=0.5
        )
        self.best_acc, self.best_state = 0.0, None

    def _step(self, batch, train=True):
        x, y = batch[0].to(DEVICE), batch[1].float().to(DEVICE)
        out  = self.model(x)
        loss = self.crit(out, y)
        if train:
            self.optim.zero_grad(); loss.backward(); self.optim.step()
        return loss.item(), out.detach().cpu().numpy(), y.cpu().numpy()

    def fit(self):
        for epoch in range(Hyper.NUM_EPOCHS):
            # ---- train ----
            self.model.train()
            for batch in self.train_loader: self._step(batch, train=True)

            # ---- val ----
            self.model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for batch in self.val_loader:
                    _, p, l = self._step(batch, train=False)
                    preds.extend(p); labels.extend(l)
            acc = accuracy_score(labels, (np.array(preds) > 0.5).astype(int))
            self.scheduler.step(acc)

            if acc > self.best_acc:
                self.best_acc, self.best_state = acc, self.model.state_dict().copy()

            if epoch - np.argmax([self.best_acc]) > 100: break  # early stop

        logger.info(f"Fold {self.fold}: best val acc={self.best_acc:.4f}")
        fname = Paths.MODELS / f"lstm_fold{self.fold}.pth"
        fname.parent.mkdir(exist_ok=True)
        torch.save(self.best_state, fname)
        return fname, self.best_acc