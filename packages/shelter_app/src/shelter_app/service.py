"""Glue between the API/CLI and the animal_shelter package."""

import io
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from animal_shelter.data import load_data
from animal_shelter.features import add_features
from animal_shelter.model.predict import load_model, predict_model
from animal_shelter.model.train import TARGET, save_model, train_model

logger = logging.getLogger("animal_shelter.app")

MODEL_SUFFIX = ".joblib"
_VALID_MODEL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class TrainResult:
    """Summary of a training run."""

    model_name: str
    model_path: Path
    n_samples: int
    classes: list[str]
    train_accuracy: float
    duration_seconds: float


def model_path_for(model_dir: Path, model_name: str) -> Path:
    """Return the path of a model, rejecting names that could escape model_dir."""
    if not _VALID_MODEL_NAME.match(model_name):
        raise ValueError(
            f"Invalid model name {model_name!r}: use only letters, digits, '_' or '-'"
        )
    return model_dir / f"{model_name}{MODEL_SUFFIX}"


def list_models(model_dir: Path) -> list[str]:
    """List the names of the models saved in model_dir."""
    return sorted(p.stem for p in model_dir.glob(f"*{MODEL_SUFFIX}"))


def records_to_raw_df(records: list[dict]) -> pd.DataFrame:
    """Load records with the raw CSV column names using `load_data`.

    The records are written to an in-memory CSV so that they go through
    exactly the same loading/cleaning steps as the CSV files on disk. Missing
    values are filled first, because a column without any value would be read
    back as float, which `load_data` cannot fill with "Unknown".
    """
    buffer = io.StringIO()
    pd.DataFrame(records).fillna("Unknown").to_csv(buffer, index=False)
    buffer.seek(0)
    return load_data(buffer)


def train_and_save(raw_df: pd.DataFrame, model_path: Path) -> TrainResult:
    """Add features, train a model on `raw_df` and save it to `model_path`."""
    if TARGET not in raw_df.columns:
        raise ValueError(f"Training data must contain the target column {TARGET!r}")

    start = time.perf_counter()
    with_features = add_features(raw_df)
    clf_model = train_model(with_features)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(clf_model, model_path)
    if not model_path.exists():
        raise RuntimeError(f"Model was not saved to {model_path}")
    duration = time.perf_counter() - start

    train_accuracy = float(
        (predict_model(clf_model, with_features) == with_features[TARGET]).mean()
    )
    logger.info(
        "Trained model on %d rows in %.2fs (train accuracy %.3f), saved to %s",
        len(raw_df),
        duration,
        train_accuracy,
        model_path,
    )
    return TrainResult(
        model_name=model_path.stem,
        model_path=model_path,
        n_samples=len(raw_df),
        classes=[str(c) for c in clf_model.classes_],
        train_accuracy=train_accuracy,
        duration_seconds=duration,
    )


def predict_df(raw_df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """Predict outcomes for `raw_df` with the model saved at `model_path`.

    Returns a DataFrame with an `id` column (taken from `animal_id`/`id` if
    present, otherwise the row number) and a `prediction` column.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("Loading model from %s", model_path)
    clf_model = load_model(model_path)
    predictions = predict_model(clf_model, add_features(raw_df))

    id_column = next((c for c in ("animal_id", "id") if c in raw_df.columns), None)
    ids = raw_df[id_column].astype(str) if id_column else raw_df.index.astype(str)
    logger.info("Generated %d predictions", len(predictions))
    return pd.DataFrame({"id": list(ids), "prediction": predictions})
