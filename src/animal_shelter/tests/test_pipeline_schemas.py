"""Tests that use Pydantic to check the output of each step of the pipeline.

Every row produced by a step is validated against a schema, so a change that
breaks the expected columns, types or values of a step fails here.
"""

from datetime import datetime
from typing import Annotated, Literal

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from animal_shelter.data import load_data
from animal_shelter.features import add_features
from animal_shelter.model import predict, train

Outcome = Literal["Adoption", "Transfer", "Return_to_owner", "Euthanasia", "Died"]


class _Row(BaseModel):
    # Fail on unexpected columns, and don't silently convert types.
    model_config = ConfigDict(extra="forbid", strict=True)


class LoadedPredictRecord(_Row):
    """A row returned by `load_data` for data in the format of test.csv."""

    id: int
    name: str
    date_time: datetime
    animal_type: Literal["Dog", "Cat"]
    sex_upon_outcome: str
    age_upon_outcome: str
    breed: str
    color: str


class LoadedTrainRecord(_Row):
    """A row returned by `load_data` for data in the format of train.csv."""

    animal_id: str
    name: str
    date_time: datetime
    outcome_type: Outcome
    outcome_subtype: str
    animal_type: Literal["Dog", "Cat"]
    sex_upon_outcome: str
    age_upon_outcome: str
    breed: str
    color: str


class _AddedFeatures(BaseModel):
    is_dog: bool
    has_name: bool
    sex: Literal["male", "female", "unknown"]
    neutered: Literal["fixed", "intact", "unknown"]
    hair_type: Literal["shorthair", "medium hair", "longhair", "unknown"]
    days_upon_outcome: Annotated[float, Field(ge=0)] | None


class TrainFeatureRecord(LoadedTrainRecord, _AddedFeatures):
    """A row returned by `add_features` for training data."""


class PredictFeatureRecord(LoadedPredictRecord, _AddedFeatures):
    """A row returned by `add_features` for data to predict on."""


def validate_rows(df: pd.DataFrame, schema: type[BaseModel]) -> list[BaseModel]:
    """Validate every row of `df` against `schema`."""
    # NaN is not a valid float for `ge=0`; it means "missing", so use None.
    records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return [schema.model_validate(record) for record in records]


@pytest.fixture(scope="module")
def train_subset_path(tmp_path_factory):
    """First 500 rows of the real training data."""
    path = tmp_path_factory.mktemp("data") / "train_subset.csv"
    pd.read_csv(train.DATA_DIR / "train.csv", nrows=500).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def test_subset_path(tmp_path_factory):
    """First 500 rows of the real test data."""
    path = tmp_path_factory.mktemp("data") / "test_subset.csv"
    pd.read_csv(train.DATA_DIR / "test.csv", nrows=500).to_csv(path, index=False)
    return path


@pytest.fixture(params=["mocked", "subset"])
def train_data_path(request, raw_data_path, train_subset_path):
    """Training data: either the mocked CSV or a subset of the real data."""
    return raw_data_path if request.param == "mocked" else train_subset_path


def test_load_data_output_schema(train_data_path):
    rows = validate_rows(load_data(train_data_path), LoadedTrainRecord)

    assert rows


def test_load_data_test_format_output_schema(test_subset_path):
    rows = validate_rows(load_data(test_subset_path), LoadedPredictRecord)

    assert len(rows) == 500


def test_add_features_output_schema(train_data_path):
    rows = validate_rows(add_features(load_data(train_data_path)), TrainFeatureRecord)

    assert rows


def test_add_features_test_format_output_schema(test_subset_path):
    rows = validate_rows(
        add_features(load_data(test_subset_path)), PredictFeatureRecord
    )

    assert len(rows) == 500


def test_train_output_schema(train_data_path, tmp_path):
    clf_model = train.train(train_data_path, tmp_path / "model.joblib")

    classes = TypeAdapter(list[Outcome]).validate_python(list(clf_model.classes_))
    assert len(classes) == len(set(classes))


def test_predict_output_schema(train_subset_path, test_subset_path, tmp_path):
    model_path = tmp_path / "model.joblib"
    train.train(train_subset_path, model_path)

    predictions = predict.predict(test_subset_path, model_path)

    TypeAdapter(list[Outcome]).validate_python(list(predictions))
    assert len(predictions) == 500


def test_schema_catches_broken_feature(raw_data_path):
    """Check that the schemas actually fail when a step returns bad values."""
    df = add_features(load_data(raw_data_path))
    df.loc[0, "sex"] = "MALE"

    with pytest.raises(ValidationError):
        validate_rows(df, TrainFeatureRecord)


def test_schema_catches_unexpected_column(raw_data_path):
    df = load_data(raw_data_path).assign(unexpected=1)

    with pytest.raises(ValidationError):
        validate_rows(df, LoadedTrainRecord)
