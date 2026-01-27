import joblib
import pandas as pd
import os

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.logger import get_logger
from src.exceptions import ModelEvaluationError

logger = get_logger("ModelEvaluation")


def evaluate_models(X_test_scaled, y_test, threshold=0.3):
    try:
        logger.info("Model evaluation started")

        models = {
            "Logistic Regression": joblib.load("models/logistic_model.pkl"),
            "Decision Tree": joblib.load("models/tree_model.pkl"),
            "XGBoost": joblib.load("models/xgb_model.pkl")
        }

        results = []

        for name, model in models.items():
            logger.info(f"Evaluating {name}")

            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            results.append({
                "Model": name,
                "Recall": recall_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "F1_Score": f1_score(y_test, y_pred),
                "ROC_AUC": roc_auc_score(y_test, y_prob),
                "PR_AUC": average_precision_score(y_test, y_prob)
            })

        results_df = pd.DataFrame(results)

        # FIX: ensure directory exists
        os.makedirs("reports", exist_ok=True)
        results_df.to_csv("reports/model_comparison.csv", index=False)

        logger.info("Model evaluation completed and saved to reports/")
        return results_df

    except Exception as e:
        logger.exception("Model evaluation failed")
        raise ModelEvaluationError("Failed during model evaluation", e)
