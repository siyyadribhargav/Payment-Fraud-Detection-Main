import joblib
import numpy as np
from src.logger import get_logger
from src.exceptions import PredictionError

logger = get_logger("FraudPredictor")


class FraudPredictor:
    def __init__(self):
        try:
            self.model = joblib.load("models/xgb_model.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
        except Exception as e:
            raise PredictionError("Failed to load model or scaler", e)

    def predict(self, input_data, threshold=0.3):
        try:
            logger.info("Prediction request received")

            input_data = np.array(input_data).reshape(1, -1)
            input_scaled = self.scaler.transform(input_data)

            if input_scaled.shape[1] != self.model.n_features_in_:
                raise ValueError("Feature count mismatch")

            fraud_prob = self.model.predict_proba(input_scaled)[0, 1]
            prediction = int(fraud_prob >= threshold)

            logger.info(
                f"Prediction done | prob={fraud_prob:.4f} | label={prediction}"
            )

            return {
                "fraud_probability": float(fraud_prob),
                "prediction": prediction
            }

        except Exception as e:
            logger.exception("Prediction failed")
            raise PredictionError("Failed during prediction", e)
