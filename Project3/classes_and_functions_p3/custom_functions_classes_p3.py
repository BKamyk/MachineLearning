from typing import Optional, List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
)


#====================================================================#
#                               Classes                              #
#====================================================================#


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop specified columns
        X_transformed = X.drop(columns=self.columns, axis=1)
        return X_transformed
    

class CustomLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str], handling_missing: List[Literal['error', 'ignore', 'infrequent_if_exist']]) -> None:
        self.handling_missing = handling_missing
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = LabelEncoder(handle_unknown=self.handling_missing)
            self.encoders[column].fit(X[column])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = self.encoders[column].transform(X[column])
        return X_transformed
    

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str], handling_missing: List[Literal['error', 'use_encoded_value']], missing_value: int = -1) -> None:
        self.missing_value = missing_value
        self.handling_missing = handling_missing
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = OrdinalEncoder(handle_unknown=self.handling_missing, unknown_value=self.missing_value)
            self.encoders[column].fit(X[[column]])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = self.encoders[column].transform(X[[column]])
        return X_transformed
    


def calculate_classification_metrics(y_valid, y_pred, classifier_name):
    """
    Calculates classification metrics for a given classifier.

    Parameters
    ----------
    y_valid : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like
        Predicted probabilities for the positive class (if applicable).
    classifier_name : str
        Name of the classifier.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing metrics such as Accuracy, Precision, Recall, F1-Score, ROC-AUC, TPR, and FPR.
    """
    # Basic metrics
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='weighted')
    recall = recall_score(y_valid, y_pred, average='weighted')
    f1 = f1_score(y_valid, y_pred, average='weighted')

    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel() if len(np.unique(y_valid)) == 2 else [None] * 4
    if tn is not None:
        tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
        fpr = fp / (fp + tn)  # False Positive Rate
    else:
        tpr, fpr = "Not applicable for multi-class", "Not applicable for multi-class"
    
    # Compile metrics into a DataFrame
    metrics = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1],
        'True Positive Rate (TPR)': [tpr],
        'False Positive Rate (FPR)': [fpr]
    }

    return pd.DataFrame(metrics, index=[classifier_name]).T


def aggregate_metrics_and_create_heatmaps_classification(metrics_list):
    """
    Aggregates a list of classification metric DataFrames into a single DataFrame and creates heatmaps for specified metrics.

    Parameters
    ----------
    metrics_list : list of pandas.DataFrame
        A list of DataFrames, each containing evaluation metrics for different classification models or datasets.

    Description
    -----------
    This function concatenates the provided list of metric DataFrames along the columns to create an aggregated DataFrame.
    It then generates heatmaps for several metrics, including 'Accuracy', 'Precision', 'Recall', 'F1-Score', and 
    'ROC-AUC'. Columns with invalid or non-numeric values are excluded from the heatmaps. The heatmaps are sorted 
    in descending order for all metrics, as higher values indicate better performance in classification.

    The heatmaps are displayed with specific formatting:
    - Metrics are plotted with a 'crest' color map.
    - All metrics use floating-point formatting with three decimal places for annotations.

    Examples
    --------
    >>> import pandas as pd
    >>> metrics_df1 = pd.DataFrame({
    ...     'Accuracy': [0.9, 0.8],
    ...     'Precision': [0.85, 0.75],
    ...     'Recall': [0.88, 0.78]
    ... }, index=['Model1', 'Model2'])
    >>> metrics_df2 = pd.DataFrame({
    ...     'Accuracy': [0.92, 0.81],
    ...     'Precision': [0.87, 0.76],
    ...     'Recall': [0.89, 0.79]
    ... }, index=['Model3', 'Model4'])
    >>> aggregate_metrics_and_create_heatmaps_classification([metrics_df1, metrics_df2])

    This will generate and display heatmaps for each specified metric.
    """

    # Aggregate metrics into a single DataFrame
    aggregated_df = pd.concat(metrics_list, axis=1)

    # Define classification metrics to plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Create heatmaps
    for metric in metrics:
        # Remove columns with invalid or non-numeric values
        valid_series = aggregated_df.loc[metric].replace(['NaN', 'Inf', 'Invalid'], np.nan).dropna()

        # Convert the series back to a DataFrame for heatmap plotting
        valid_df = valid_series.to_frame().sort_values(by=metric, ascending=False)

        plt.figure(figsize=(10, 6))
        sns.heatmap(valid_df.astype(float), annot=True, cmap='crest',
                    cbar=True, linewidth=.5, fmt='.6f', annot_kws={"size": 15})
        plt.title(f'{metric}')
        plt.show()