import pandas as pd
from src.logger import get_logger
from src.exceptions import DataLoadingError

logger = get_logger("DataPreprocessor")


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            logger.info("Loading raw data")
            df = pd.read_csv(self.file_path)
            df = df.drop_duplicates()
            logger.info(f"Raw data loaded | Shape: {df.shape}")
            return df
        except Exception as e:
            logger.exception("Failed during data loading")
            raise DataLoadingError("Error while loading raw data", e)
