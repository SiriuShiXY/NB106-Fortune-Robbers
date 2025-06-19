"""
store 5-fold LSTM to progress.pkl
"""
import pickle, os
from pathlib import Path
from tqdm import tqdm
from config import Paths, logger
from data.split import make_splits             # 需在 split.py 实现
from data.preprocess import preprocess
from trainers.lstm_trainer import LSTMTrainer

PROGRESS = Paths.BASE / "progress.pkl"
done = pickle.load(open(PROGRESS, "rb")) if PROGRESS.exists() else set()

for csv in tqdm(os.listdir(Paths.CLEAN), desc="Files"):
    if not csv.endswith(".csv") or csv in done: continue
    splits = make_splits(Paths.CLEAN / csv)

    for fold, (tr, val, _) in enumerate(splits):
        Xtr = preprocess(tr)
        Xva = preprocess(val)
        if Xtr[0].size == 0 or Xva[0].size == 0:
            logger.warning(f"{csv} fold{fold}: empty after preprocess."); continue
        trainer = LSTMTrainer(Xtr, Xva, fold=fold+1)
        trainer.fit()

    done.add(csv); pickle.dump(done, open(PROGRESS, "wb"))
    logger.info(f"{csv} finished ✓")