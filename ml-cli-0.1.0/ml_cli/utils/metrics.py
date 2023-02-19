from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from ml_cli.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate(y_true, y_pred):
    """
    Evaluate a binary classification model using common metrics.
    """
    logger.debug('Evaluating model')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return {'accuracy': round(accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1': round(f1, 2),
            'auc': round(auc ,2)}

def cross_validate(X, y, model, cv=5, scoring='accuracy'):
    """
    Cross validate a binary classification model using common metrics.
    """
    logger.debug('Cross validating model')
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores