"""Tests that the model's performance doesn't fall below fixed thresholds.

The model is trained on 80% of the real training data and evaluated on the
remaining 20%. The thresholds sit a little below the scores measured when
these tests were written (accuracy 0.586, macro recall 0.382, macro precision
0.43, training in ~0.35s), so they catch real regressions without failing on
small fluctuations.
"""

import time

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from animal_shelter.data import load_data
from animal_shelter.features import add_features
from animal_shelter.model import predict, train

MIN_ACCURACY = 0.55
MIN_IMPROVEMENT_OVER_BASELINE = 0.10
MIN_MACRO_RECALL = 0.35
MIN_MACRO_PRECISION = 0.38
MIN_ADOPTION_RECALL = 0.65

# Generous limits so the tests don't fail on a slower machine.
MAX_TRAIN_SECONDS = 10.0
MAX_PREDICT_SECONDS = 2.0


@pytest.fixture(scope="module")
def train_test_data():
    df = add_features(load_data(train.DATA_DIR / "train.csv"))
    return train_test_split(
        df, test_size=0.2, stratify=df[train.TARGET], random_state=42
    )


@pytest.fixture(scope="module")
def trained(train_test_data):
    """Train the model once, returning it with its training time in seconds."""
    train_df, _ = train_test_data
    start = time.perf_counter()
    clf_model = train.train_model(train_df)
    return clf_model, time.perf_counter() - start


@pytest.fixture(scope="module")
def evaluation(trained, train_test_data):
    """Predict on the held-out data, returning (y_true, y_pred, seconds)."""
    clf_model, _ = trained
    _, test_df = train_test_data
    start = time.perf_counter()
    y_pred = predict.predict_model(clf_model, test_df)
    return test_df[train.TARGET], y_pred, time.perf_counter() - start


def test_accuracy(evaluation):
    y_true, y_pred, _ = evaluation

    assert accuracy_score(y_true, y_pred) >= MIN_ACCURACY


def test_better_than_baseline(evaluation, train_test_data):
    y_true, y_pred, _ = evaluation
    train_df, test_df = train_test_data
    x_train, y_train = train.split_features_target(train_df)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    baseline_accuracy = accuracy_score(
        y_true, baseline.predict(train.select_features(test_df))
    )

    assert accuracy_score(y_true, y_pred) >= (
        baseline_accuracy + MIN_IMPROVEMENT_OVER_BASELINE
    )


def test_macro_recall(evaluation):
    y_true, y_pred, _ = evaluation

    assert recall_score(y_true, y_pred, average="macro") >= MIN_MACRO_RECALL


def test_macro_precision(evaluation):
    y_true, y_pred, _ = evaluation

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    assert precision >= MIN_MACRO_PRECISION


def test_adoption_recall(evaluation):
    """The most common outcome should be recognised most of the time."""
    y_true, y_pred, _ = evaluation

    recall = recall_score(y_true, y_pred, labels=["Adoption"], average=None)[0]
    assert recall >= MIN_ADOPTION_RECALL


def test_training_time(trained):
    _, seconds = trained

    assert seconds <= MAX_TRAIN_SECONDS


def test_prediction_time(evaluation):
    _, _, seconds = evaluation

    assert seconds <= MAX_PREDICT_SECONDS
