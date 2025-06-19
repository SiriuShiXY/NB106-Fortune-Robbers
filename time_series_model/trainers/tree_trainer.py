import numpy as np
from sklearn.metrics import accuracy_score
from models.tree_models import get_rf, get_gbdt, get_lgbm
from config import Hyper, logger
from metrics import evaluate_classification

MODELS = dict(rf=get_rf, gbdt=get_gbdt, lgbm=get_lgbm)

def train_tree(model_name, X_train, y_train, X_val, y_val):
    mdl = MODELS[model_name]()
    Xtr2 = X_train.reshape(len(y_train), -1)
    Xva2 = X_val.reshape(len(y_val),   -1)
    mdl.fit(Xtr2, y_train)

    preds = mdl.predict_proba(Xva2)[:, 1]
    metric = evaluate_classification(preds, y_val)
    logger.info(f"{model_name.upper()}  val-acc={metric['Accuracy']:.4f}")
    return mdl, metric