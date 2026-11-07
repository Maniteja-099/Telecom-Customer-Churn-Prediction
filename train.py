"""Train the churn model and save evaluation metrics and the fitted pipeline."""
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import DATA_PATH, load_data
from src.model import build_pipeline, split_features


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "churn_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Dataset not found. Run: python generate_data.py")
    frame = load_data()
    features, target = split_features(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = build_pipeline(x_train)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "rows": int(len(frame)),
        "churn_rate": float(target.mean()),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "classification_report": classification_report(y_test, predicted, output_dict=True),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.3f} | ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
