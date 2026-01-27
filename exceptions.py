class FraudDetectionException(Exception):
    """
    Base exception class for Fraud Detection project
    """
    def __init__(self, message, cause=None):
        super().__init__(message)
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{self.__class__.__name__}: {self.args[0]} | Cause: {self.cause}"
        return f"{self.__class__.__name__}: {self.args[0]}"


class DataLoadingError(FraudDetectionException):
    pass


class FeatureEngineeringError(FraudDetectionException):
    pass


class ModelTrainingError(FraudDetectionException):
    pass


class ModelEvaluationError(FraudDetectionException):
    pass


class PredictionError(FraudDetectionException):
    pass
