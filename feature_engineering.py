import pandas as pd
from src.logger import get_logger
from src.exceptions import FeatureEngineeringError

logger = get_logger("FeatureEngineer")


class FeatureEngineer:
    def __init__(self, dataframe):
        self.df = dataframe

    def create_features(self):
        try:
            logger.info("Starting feature engineering")
            
            df = self.df.copy()
            df['hour'] = df['step'] % 24
            df['day'] = df['step'] // 24
            df['merchant'] = df['nameDest'].str.startswith('M').astype('int8')

            df['orgdiffbalance'] = df['oldbalanceOrg'] - df['newbalanceOrig']
            df['destdiffbalance'] = df['oldbalanceDest'] - df['newbalanceDest']

            df = pd.get_dummies(df, columns=['type'])

            drop_cols = [
                'step','nameOrig','nameDest',
                'type_CASH_IN','type_DEBIT',
                'type_PAYMENT','isFlaggedFraud'
            ]
            df.drop(columns=drop_cols, inplace=True)

            df = df.astype('int')
            #after extraction all feature we will store the csv file in processed folder as processed data
            df.to_csv("data/processed/processed_data.csv", index=False)
            logger.info("Processed data saved")

            return df

        except Exception as e:
            logger.exception("Feature engineering failed")
            raise FeatureEngineeringError("Error during feature engineering", e)
