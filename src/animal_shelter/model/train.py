"""Train a model on the animal shelter data."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from animal_shelter.data import load_data
from animal_shelter.features import add_features

CAT_FEATURES: list[str] = [
    "animal_type",
    "is_dog",
    "has_name",
    "sex",
    "hair_type",
]
NUM_FEATURES: list[str] = ["days_upon_outcome"]
TARGET: str = "outcome_type"
RANDOM_STATE: int = 28

# src/animal_shelter/model/train.py -> project root
DATA_DIR: Path = Path(__file__).resolve().parents[3] / "data"
DEFAULT_MODEL_PATH: Path = DATA_DIR / "model.joblib"


def build_transformer(
    cat_features: list[str] = CAT_FEATURES, num_features: list[str] = NUM_FEATURES
) -> ColumnTransformer:
    """Build the preprocessing transformer for numeric and categorical features.

    Parameters
    ----------
    cat_features : list of str
        Names of the categorical feature columns
    num_features : list of str
        Names of the numeric feature columns

    Returns
    -------
    transformer : sklearn.compose.ColumnTransformer
        Transformer that imputes and scales numeric features and
        one-hot encodes categorical features

    """
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
    )
    cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])
    transformer = ColumnTransformer(
        (
            ("numeric", num_transformer, num_features),
            ("categorical", cat_transformer, cat_features),
        )
    )
    return transformer


def build_model(
    cat_features: list[str] = CAT_FEATURES,
    num_features: list[str] = NUM_FEATURES,
    random_state: int | None = RANDOM_STATE,
) -> Pipeline:
    """Build the full model pipeline (preprocessing + classifier).

    Parameters
    ----------
    cat_features : list of str
        Names of the categorical feature columns
    num_features : list of str
        Names of the numeric feature columns
    random_state : int or None
        Seed for the classifier, so that training is reproducible

    Returns
    -------
    clf_model : sklearn.pipeline.Pipeline
        Unfitted model pipeline

    """
    transformer = build_transformer(cat_features, num_features)
    clf_model = Pipeline(
        [
            ("transformer", transformer),
            ("model", RandomForestClassifier(random_state=random_state)),
        ]
    )
    return clf_model


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


def split_features_target(
    df: pd.DataFrame,
    cat_features: list[str] = CAT_FEATURES,
    num_features: list[str] = NUM_FEATURES,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features (see add_features)
    cat_features : list of str
        Names of the categorical feature columns
    num_features : list of str
        Names of the numeric feature columns
    target : str
        Name of the target column

    Returns
    -------
    x : pandas.DataFrame
        Feature columns
    y : pandas.Series
        Target column

    """
    x = select_features(df, cat_features, num_features)
    y = df[target]
    return x, y


def train_model(df: pd.DataFrame) -> Pipeline:
    """Build and fit a model on a DataFrame with features.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features (see add_features)

    Returns
    -------
    clf_model : sklearn.pipeline.Pipeline
        Fitted model pipeline

    """
    x, y = split_features_target(df)
    clf_model = build_model()
    clf_model.fit(x, y)
    return clf_model


def save_model(clf_model: Pipeline, path: str | Path) -> None:
    """Save a fitted model to disk.

    Parameters
    ----------
    clf_model : sklearn.pipeline.Pipeline
        Fitted model pipeline
    path : str or pathlib.Path
        Path to write the model to

    """
    joblib.dump(clf_model, path)


def train(
    data_path: str | Path = DATA_DIR / "train.csv",
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """Load the training data, train a model and save it."""
    raw_data = load_data(data_path)
    with_features = add_features(raw_data)
    clf_model = train_model(with_features)
    save_model(clf_model, model_path)
    return clf_model
