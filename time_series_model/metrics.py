import numpy as np, torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.stats import spearmanr

def evaluate_rankic(preds, targs, n_sec=478):
    rank_ic, _ = spearmanr(preds, targs)
    daily = [spearmanr(preds[i*n_sec:(i+1)*n_sec],
                       targs[i*n_sec:(i+1)*n_sec])[0]
             for i in range(len(preds)//n_sec)]
    rank_icir = np.nan if len(daily)<2 else np.mean(daily)/np.std(daily)
    return rank_ic, rank_icir

def evaluate_classification(preds, targs):
    bin_t = (targs > 0.5).astype(int)
    bin_p = (preds >= 0.5).astype(int)
    return dict(
        Accuracy = accuracy_score(bin_t, bin_p),
        Precision= precision_score(bin_t, bin_p, zero_division=0),
        Recall   = recall_score(bin_t, bin_p, zero_division=0),
        AUC      = roc_auc_score(bin_t, preds) if len(np.unique(bin_t))>1 else np.nan,
    )