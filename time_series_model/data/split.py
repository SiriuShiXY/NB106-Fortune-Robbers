"""
5 fold time series split for stock data
"""
import pandas as pd
from datetime import datetime
from config import logger, Hyper

# 5 periods with start and end dates
_BOUNDS = [
    ("2015/10/26", "2018/12/21"),
    ("2017/5/05",  "2020/08/16"),
    ("2018/12/19", "2022/04/01"),
    ("2020/08/13", "2023/11/25"),
    ("2021/03/14", "2024/06/25"),
]
_BOUNDS = [(datetime.strptime(s, "%Y/%m/%d"),
            datetime.strptime(e, "%Y/%m/%d")) for s, e in _BOUNDS]

def _split_one(df: pd.DataFrame):
    """ 7:2:1  train / val / test"""
    q70 = df['日期'].quantile(0.7)
    q90 = df['日期'].quantile(0.9)
    train = df[df['日期'] < q70]
    val   = df[(df['日期'] >= q70) & (df['日期'] < q90)]
    test  = df[df['日期'] >= q90]
    return train, val, test

def make_splits(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['日期'])
    parts = []
    for (s, e) in _BOUNDS:
        part = df[(df['日期'] >= s) & (df['日期'] <= e)]
        if part.empty:
            logger.warning(f"{csv_path.name}: no rows between {s} ~ {e}")
            parts.append((None, None, None))
        else:
            parts.append(_split_one(part))
    return parts  # [(train,val,test)*5]