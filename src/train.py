import pickle
import logging
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from model import RestaurantSuccessModel
from utils import plot_confusion_matrix, plot_feature_importance, plot_roc_curve, print_evaluation_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_preprocessed_data(data_path='../data/preprocessed/'):
    required_files = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl', 
                      'y_test.pkl', 'feature_names.pkl']
    
    for filename in required_files:
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Required file not found: {filepath}\nHave you run data_pipeline.py first?"
            )
    
    logger.info(f"Loading preprocessed data from {data_path}...")
    
    with open(os.path.join(data_path, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(data_path, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(data_path, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(data_path, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    with open(os.path.join(data_path, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    logger.info(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names


def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model.")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics, y_pred, y_pred_proba


def cross_validate_model(model, X_train, y_train, cv=5):
    logger.info(f"Performing {cv}-fold cross-validation...")
    cv_scores = cross_val_score(model.model, X_train, y_train, cv=cv, scoring='roc_auc')
    logger.info(f"CV ROC-AUC scores: {cv_scores}")
    logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return cv_scores


def save_model(model, filepath='restaurant_model.pkl'):
    folder = os.path.dirname(filepath) or '.'  
    os.makedirs(folder, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {filepath}")


def try_multiple_models(X_train, y_train, cv=5, scoring='roc_auc'):

    model_types = ['random_forest', 'gradient_boosting', 'decision_tree', 'logistic']
    
    best_model = None
    best_score = -np.inf
    model_scores = {}
    
    for model_type in model_types:
        logger.info(f"Initializing {model_type} model...")
        model_instance = RestaurantSuccessModel(model_type=model_type)
        scores = cross_val_score(model_instance.model, X_train, y_train, cv=cv, scoring=scoring)
        mean_score = scores.mean()
        model_scores[model_type] = mean_score
        logger.info(f"{model_type} mean CV {scoring}: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model_instance
    
    logger.info(f"Selected best model: {best_model.model_type} with CV {scoring}: {best_score:.4f}")
    
    # Train best model on full training set
    best_model.fit(X_train, y_train)
    
    return best_model, model_scores


if __name__ == "__main__":
    
    # Create output directories
    os.makedirs('../outputs', exist_ok=True)    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data()
    
    # Try multiple models and pick the best
    best_model, all_scores = try_multiple_models(X_train, y_train)
    
    # Cross-validation (optional)
    cross_validate_model(best_model, X_train, y_train)
    
    # Evaluate on test set
    metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test)
    
    # Print detailed metrics
    print_evaluation_metrics(y_test, y_pred, y_pred_proba)
    
    # Plot results
    plot_confusion_matrix(y_test, y_pred, save_path='../outputs/confusion_matrix.png')
    plot_roc_curve(y_test, y_pred_proba, save_path='../outputs/roc_curve.png')
    plot_feature_importance(best_model.model, feature_names, top_n=20, save_path='../outputs/feature_importance.png')
    
    # Save best model in same directory
    save_model(best_model, '../models/restaurant_model.pkl')
    
    logger.info("Training pipeline complete")
