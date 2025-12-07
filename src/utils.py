import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()  
        else:
            plt.show()
    else:
        logger.warning(f"Model does not have feature_importances_ attribute")


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def print_evaluation_metrics(y_true, y_pred, y_pred_proba=None):
    print("EVALUATION METRICS")
    print("="*60)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Unsuccessful', 'Successful']))
    
    if y_pred_proba is not None:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC-AUC Score: {auc:.4f}")
    
    print("="*60)