import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from config import Hyper, logger

def preprocess(df: pd.DataFrame):
    """
    Preprocess the DataFrame for time series modeling.
    """
    dates, codes = df['日期'].values, df['代码'].values
    target       = df['rank_if'].values

    # only keep numeric columns
    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df.fillna(method='ffill', inplace=True)
    num_df.fillna(method='bfill', inplace=True)

    scaler   = StandardScaler()
    features = scaler.fit_transform(num_df)
    features = np.hstack([features, target.reshape(-1, 1)])

    if len(features) < Hyper.LAG_PERIOD:
        logger.warning("Not enough rows (< lag); return empty.")
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    lagged, lag_dates, lag_codes = [], [], []
    for i in range(Hyper.LAG_PERIOD, len(features)):
        lagged.append(features[i-Hyper.LAG_PERIOD:i])
        lag_dates.append(dates[i])
        lag_codes.append(codes[i])

    return (
        np.array(lagged).astype(np.float32),
        target[Hyper.LAG_PERIOD:].astype(np.float32),
        np.array(lag_dates),
        np.array(lag_codes)
    )