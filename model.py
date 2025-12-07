from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RestaurantSuccessModel:
    
    def __init__(self, model_type='random_forest', **kwargs):
        self.model_type = model_type
        self.model = self._get_model(**kwargs)
        logger.info(f"Initialized {model_type} model")
    
    def _get_model(self, **kwargs):
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 15),
                min_samples_split=kwargs.get('min_samples_split', 10),
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42
            )
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 20),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        logger.info(f"Training {self.model_type}.")
        self.model.fit(X, y)
        logger.info("Training complete")
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])  # Absolute coefficients
        else:
            logger.warning(f"{self.model_type} doesn't support feature importance")
        return None


# Model hyperparameters configurations
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10
    },
    'logistic': {
        'max_iter': 1000
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    }
}