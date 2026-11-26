"""Make predictions with a trained animal shelter model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from animal_shelter.data import load_data
from animal_shelter.features import add_features
from animal_shelter.model.train import DATA_DIR, DEFAULT_MODEL_PATH, select_features

__all__ = ["load_model", "predict", "predict_model", "select_features"]


def load_model(path: str | Path) -> Pipeline:
    """Load a fitted model from disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the saved model

    Returns
    -------
    clf_model : sklearn.pipeline.Pipeline
        Fitted model pipeline

    """
    return joblib.load(path)


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
    data_path: str | Path = DATA_DIR / "test.csv",
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> np.ndarray:
    """Load the test data and a trained model, and make predictions."""
    test_data = load_data(data_path)
    with_features = add_features(test_data)
    clf_model = load_model(model_path)
    predictions = predict_model(clf_model, with_features)
    return predictions
