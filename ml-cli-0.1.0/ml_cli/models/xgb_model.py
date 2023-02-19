import joblib
import pandas as pd
import xgboost as xgb
from ml_cli.utils.logger import get_logger

logger = get_logger(__name__)
class TitanicXGBModel:
    
    def __init__(self, params):
        self.params = params
        self.model = None
        self.feature_names = None
        self.target_name = None
        self.feature_importance = None
        
    def fit(self, X_train, y_train):
        logger.debug("Fitting XGBoost model")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.target_name = y_train.name
        self.feature_names = X_train.columns
        self.model = xgb.train(self.params, dtrain)
    
    def save(self, filename):
        logger.debug(f"Saving XGBoost model to {filename}")
        joblib.dump(self, filename)

    def predict(self, X_test, probability=False):
        logger.debug("Predicting with XGBoost model")
        if self.model is None:
            logger.error("XGBoost model is not trained yet")
            raise ValueError("XGBoost model is not trained yet")
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        if probability:
            return pd.Series(y_pred, name=self.target_name)
        else:
            return pd.Series(y_pred.round(), name=self.target_name)

    def get_feature_importance(self):
        logger.debug("Getting feature importance from XGBoost model")
        if self.model is None or self.feature_names is None:
            logger.error("XGBoost model is not trained yet or feature names are not available")
            raise ValueError("XGBoost model is not trained yet or feature names are not available")
        importance = self.model.get_score(importance_type='gain')
        importance = pd.Series(importance, name='importance')
        importance.index.name = 'feature'
        importance = importance.reset_index()
        importance = importance.sort_values('importance', ascending=False)
        importance['importance'] = importance['importance'] / importance['importance'].sum()
        importance = importance.set_index('feature')
        importance = importance['importance']
        importance = importance.reindex(self.feature_names, fill_value=0)
        return importance
