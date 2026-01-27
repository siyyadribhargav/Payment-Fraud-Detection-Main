#for saving the trained models in pkl files
import joblib

#train test split of the models
from sklearn.model_selection import train_test_split

#oversampling the minority data
from imblearn.over_sampling import SMOTE

#this is for scalin the dataset 
from sklearn.preprocessing import StandardScaler

#models that we will use in the training the dataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.logger import get_logger
from src.exceptions import ModelTrainingError

logger = get_logger("ModelTrainer")


class ModelTrainer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def train(self):
        try:
            logger.info("Model training started")

            #seperating the feature isfraud from the cleaned dataset
            X = self.dataframe.drop(columns=['isFraud'])
            Y = self.dataframe['isFraud']

            #here i have splitted the datasetr into train and test
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42, stratify=Y
            )

            logger.info("Train-test split completed")

            #here we have oversample the data of minority class that isFraud ==1
            smotr = SMOTE(random_state=42)
            X_train_over, Y_train_over = smotr.fit_resample(X_train, Y_train)

            logger.info("SMOTE oversampling applied")

            #scaling the dataset
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_over)
            X_test_scaled = scaler.transform(X_test)

            logger.info("Feature scaling completed")

            ##upto her we have completed the steps needed to do before training the models

            #here we call the models together 

            logistic_params = {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
            }

            logistic = LogisticRegression(
                class_weight='balanced',
                max_iter=500,
            )

            logger.info("Training Logistic Regression")

            logistic_search = RandomizedSearchCV(
                logistic,
                logistic_params,
                n_iter=5,
                random_state=42,
                scoring='recall',
                cv=3,
                n_jobs=2
            )

            logistic_search.fit(X_train_scaled, Y_train_over)

            logger.info("Logistic Regression training completed")

            tree_params = {
                "max_depth": [6, 8, 10],
                "min_samples_leaf": [20, 50, 100],
                "min_samples_split": [50, 100],
                "class_weight": ["balanced"]
            }

            tree = DecisionTreeClassifier(
                class_weight='balanced',
                random_state=42
            )

            logger.info("Training Decision Tree")

            tree_search = RandomizedSearchCV(
                tree,
                tree_params,
                n_iter=10,
                random_state=42,
                scoring='recall',
                cv=3,
                n_jobs=2
            )

            tree_search.fit(X_train_scaled, Y_train_over)

            logger.info("Decision Tree training completed")

            xgb_params = {
                "n_estimators": [50, 100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }

            xgb = XGBClassifier(
                scale_pos_weight=(Y_train_over == 0).sum() / (Y_train_over == 1).sum(),
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=2
            )

            logger.info("Training XGBoost")

            xgb_search = RandomizedSearchCV(
                xgb,
                xgb_params,
                n_iter=10,
                random_state=42,
                scoring='recall',
                cv=3,
                n_jobs=2
            )

            xgb_search.fit(X_train_scaled, Y_train_over)

            logger.info("XGBoost training completed")

            #params are the hyperparameter tunning for the models
            #best_params gives the best parameters after tunnign

            joblib.dump(logistic_search.best_estimator_, "models/logistic_model.pkl")
            joblib.dump(tree_search.best_estimator_, "models/tree_model.pkl")
            joblib.dump(xgb_search.best_estimator_, "models/xgb_model.pkl")
            joblib.dump(scaler, "models/scaler.pkl")

            logger.info("Models and scaler saved successfully")

            return X_test, X_test_scaled, Y_test

        except Exception as e:
            logger.exception("Error occurred during model training")
            raise ModelTrainingError("Model training failed", e)
