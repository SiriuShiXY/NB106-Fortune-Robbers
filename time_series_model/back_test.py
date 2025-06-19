"""
Based on the Top-K prediction, simulate a backtest
"""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt, statsmodels.api as sm
from config import logger

def _load_index(index_path: Path):
    idx = pd.read_excel(index_path).rename(columns={'日期':'date','收盘价':'close'})
    idx['date'] = pd.to_datetime(idx['date']).dt.strftime('%Y-%m-%d')
    idx.set_index('date', inplace=True)
    if '%Chg' not in idx.columns:
        idx['%Chg'] = idx['close'].pct_change()*100
    return idx

def run_backtest(topk_csv: Path, data_dir: Path,
                 index_path: Path, initial=1_000_000,
                 trading_fee=3e-4, k_sec=50):
    idx_df = _load_index(index_path)
    df = pd.read_csv(topk_csv)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    groups = df.groupby('date')['code'].apply(list)

    capital, daily_ret, idx_ret = initial, [], []
    for cur, nxt in zip(groups.index[:-1], groups.index[1:]):
        stocks = groups[cur][:k_sec]
        profit = 0.
        for code in stocks:
            path = data_dir / f"{code}_ors.csv"
            try:
                sdf = pd.read_csv(path)
            except FileNotFoundError:
                continue
            sdf['日期'] = pd.to_datetime(sdf['日期']).dt.strftime('%Y-%m-%d')
            row = sdf[sdf['日期'] == cur].index
            if not row.size or row[0]+1 >= len(sdf): continue
            prev, next_ = sdf.loc[row[0], '均价(元)'], sdf.loc[row[0]+1, '均价(元)']
            if prev == 0 or next_ == 0: continue
            shares = capital/len(stocks)/prev
            profit += shares*(next_-prev) - shares*prev*trading_fee
        daily_ret.append(profit/capital*100)
        capital += profit
        idx_ret.append(idx_df.loc[nxt,'%Chg'] if nxt in idx_df.index else 0)

    # -------- metrics --------
    port, mkt = pd.Series(daily_ret), pd.Series(idx_ret)
    sharpe_p  = (port.mean()/port.std())*np.sqrt(252)
    sharpe_m  = (mkt.mean()/mkt.std())*np.sqrt(252)
    logger.info(f"{topk_csv.name}: total={capital/initial-1:.2%} sharpe={sharpe_p:.2f}")

    # -------- plot --------
    cp = np.cumprod(1+port/100)
    cm = np.cumprod(1+mkt /100)
    plt.plot(cp, label='Portfolio'); plt.plot(cm, label='CSI300')
    plt.title(topk_csv.name); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # CAPM
    rf = 0.018/252
    excess_p, excess_m = port/100 - rf, mkt/100 - rf
    capm = sm.OLS(excess_p.dropna(), sm.add_constant(excess_m.dropna())).fit()
    logger.info(f"CAPM  α_ann={capm.params['const']*252:.2%}  β={capm.params['mkt']:.2f}")