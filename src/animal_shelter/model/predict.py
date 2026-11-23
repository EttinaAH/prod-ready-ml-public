"""Make predictions with a trained animal shelter model."""

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from animal_shelter.data import load_data
from animal_shelter.features import add_features
from animal_shelter.model.train import CAT_FEATURES, NUM_FEATURES


def load_model(path: str) -> Pipeline:
    """Load a fitted model from disk.

    Parameters
    ----------
    path : str
        Path to the saved model

    Returns
    -------
    clf_model : sklearn.pipeline.Pipeline
        Fitted model pipeline

    """
    return joblib.load(path)


def select_features(
    df: pd.DataFrame,
    cat_features: list[str] = CAT_FEATURES,
    num_features: list[str] = NUM_FEATURES,
) -> pd.DataFrame:
    """Select the feature columns used by the model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features (see add_features)
    cat_features : list of str
        Names of the categorical feature columns
    num_features : list of str
        Names of the numeric feature columns

    Returns
    -------
    x : pandas.DataFrame
        Feature columns

    """
    return df[cat_features + num_features]


def predict_model(clf_model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predict outcomes for a DataFrame with features.

    Parameters
    ----------
    clf_model : sklearn.pipeline.Pipeline
        Fitted model pipeline
    df : pandas.DataFrame
        DataFrame with features (see add_features)

    Returns
    -------
    predictions : numpy.ndarray
        Predicted outcome types

    """
    x = select_features(df)
    return clf_model.predict(x)


def predict(
    data_path: str = "../data/test.csv", model_path: str = "../data/model.joblib"
) -> np.ndarray:
    """Load the test data and a trained model, and make predictions."""
    test_data = load_data(data_path)
    with_features = add_features(test_data)
    clf_model = load_model(model_path)
    predictions = predict_model(clf_model, with_features)
    return predictions
