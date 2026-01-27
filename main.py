from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer
from src.evaluate_models import evaluate_models
from src.logger import get_logger


logger = get_logger("MainPipeline")


def main():
    try:
        logger.info("Fraud Detection Pipeline Started")

        df = DataPreprocessor("data/raw/data_table.csv").load_data()
        df = FeatureEngineer(df).create_features()

        trainer = ModelTrainer(df)
        X_test, X_test_scaled, y_test = trainer.train()

        results = evaluate_models(X_test_scaled, y_test, threshold=0.3)
        print(results)

        logger.info("Fraud Detection Pipeline Completed Successfully")

    except Exception as e:
        logger.exception("Pipeline failed")
        raise e


if __name__ == "__main__":
    main()
