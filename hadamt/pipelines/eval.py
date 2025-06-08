from sklearn.metrics import roc_auc_score
import json
from pathlib import Path


def save_report(y_true, scores, path="results/report.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    auc = roc_auc_score(y_true, scores)
    with open(path, "w") as f:
        json.dump({"roc_auc": auc}, f)
    return auc
