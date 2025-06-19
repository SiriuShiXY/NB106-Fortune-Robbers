import argparse, torch, numpy as np
from pathlib import Path
from config import Paths, Hyper, logger
from data.preprocess import preprocess
from datasets.stock_dataset import StockDataset
from torch.utils.data import DataLoader
from models.lstm import LSTMModel
from data.io import save_df

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_lstm(model_path: Path, input_size: int):
    state_dict = torch.load(model_path, map_location=DEVICE)
    mdl = LSTMModel(input_size)
    mdl.load_state_dict(state_dict)
    mdl.eval().to(DEVICE)
    return mdl

def make_topk(csv_path: Path, model_path: Path, fold: int):
    df = preprocess_full_csv(csv_path)  # define as needed
    X, _, dates, codes = df
    loader = DataLoader(StockDataset(X, np.zeros(len(X)), dates, codes),
                        batch_size=Hyper.BATCH_SIZE)

    mdl = load_lstm(model_path, input_size=X.shape[-1])

    probs, dt, cd = [], [], []
    with torch.no_grad():
        for x, _, d, c in loader:
            p = mdl(x.to(DEVICE)).cpu().numpy()
            probs.extend(p); dt.extend(d); cd.extend(c)

    out = (np.rec.fromarrays([dt, cd, probs],
            names=['date','code','probability']))
    # ------- Top-K -------
    topk_df = (np.rec.array(out)
               .groupby('date', size=Hyper.K_TOP_STOCKS,
                        key='probability', reverse=True))
    save_df(topk_df, Paths.TOPK / f"top_k_stocks_per_day_fold{fold}.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--fold", type=int, default=1)
    args = ap.parse_args()
    make_topk(Path(args.csv), Path(args.model), args.fold)