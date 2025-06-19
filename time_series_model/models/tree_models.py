from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier

def get_rf(**kw):   return RandomForestClassifier(n_estimators=100, n_jobs=-1, **kw)
def get_gbdt(**kw): return GradientBoostingClassifier(n_estimators=100, **kw)
def get_lgbm(**kw): return LGBMClassifier(n_estimators=100, verbose=-1, **kw)