"""End-to-end tests: load data, train, predict and serve the predictions.

Each test runs the full chain through the CLI and the API, once with mocked
data and once with a subset of the real training data.
"""

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from shelter_app.api import app as api
from shelter_app.cli import app as cli
from shelter_app.schemas import PredictResponse, TrainResponse
from sklearn.metrics import accuracy_score
from typer.testing import CliRunner

from animal_shelter.model.train import DATA_DIR

runner = CliRunner()

OUTCOMES = ["Adoption", "Transfer", "Return_to_owner", "Euthanasia", "Died"]

# The model trained on 2000 real rows scores ~0.57 on the held-out rows;
# always predicting the most common outcome scores ~0.40.
MIN_SUBSET_ACCURACY = 0.5
MAX_API_PREDICT_SECONDS = 2.0


def make_mocked_train_df(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate mocked data in the raw format of data/train.csv."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "AnimalID": [f"A{i}" for i in range(n_rows)],
            "Name": rng.choice(["Ivo", "Bella", "Henk", None], n_rows),
            "DateTime": pd.Timestamp("2014-01-01")
            + pd.to_timedelta(rng.integers(0, 365 * 24, n_rows), unit="h"),
            "OutcomeType": rng.choice(OUTCOMES, n_rows),
            "OutcomeSubtype": rng.choice(["Partner", "Foster", None], n_rows),
            "AnimalType": rng.choice(["Dog", "Cat"], n_rows),
            "SexuponOutcome": rng.choice(
                ["Neutered Male", "Spayed Female", "Intact Male", "Unknown"], n_rows
            ),
            "AgeuponOutcome": [
                f"{rng.integers(1, 15)} {rng.choice(['weeks', 'months', 'years'])}"
                for _ in range(n_rows)
            ],
            "Breed": rng.choice(
                ["Domestic Shorthair Mix", "Labrador Retriever Mix", "Pit Bull"],
                n_rows,
            ),
            "Color": rng.choice(["Black", "White"], n_rows),
        }
    )


def to_predict_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Turn data in the train.csv format into the test.csv format."""
    return (
        train_df.drop(columns=["OutcomeType", "OutcomeSubtype"])
        .rename(columns={"AnimalID": "ID"})
        .reset_index(drop=True)
    )


def to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a raw DataFrame into JSON records for the API."""
    df = df.astype({"DateTime": str})
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def run_cli(*args) -> str:
    result = runner.invoke(cli, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output
    return result.output


@pytest.fixture(params=["mocked", "subset"])
def dataset(request):
    """Return (train_df, holdout_df) in the raw train.csv format."""
    if request.param == "mocked":
        df = make_mocked_train_df(300, seed=0)
    else:
        df = pd.read_csv(DATA_DIR / "train.csv").sample(2500, random_state=0)
    n_holdout = len(df) // 5
    return df.iloc[:-n_holdout], df.iloc[-n_holdout:]


def test_end_to_end_cli_and_api(dataset, tmp_path, model_dir):
    """Train with the CLI, then predict with both the CLI and the API."""
    train_df, holdout_df = dataset
    predict_df = to_predict_df(holdout_df)
    train_path, predict_path = tmp_path / "train.csv", tmp_path / "predict.csv"
    train_df.to_csv(train_path, index=False)
    predict_df.to_csv(predict_path, index=False)

    # Train and predict with the CLI.
    model_path = model_dir / "e2e.joblib"
    run_cli("train", train_path, model_path)
    output_path = tmp_path / "predictions.csv"
    run_cli("predict", predict_path, model_path, output_path)
    cli_predictions = pd.read_csv(output_path, dtype={"id": str})

    assert list(cli_predictions["id"]) == list(predict_df["ID"].astype(str))
    assert set(cli_predictions["prediction"]) <= set(train_df["OutcomeType"])

    # Serve predictions from the same model through the API.
    client = TestClient(api)
    assert client.get("/models/").json() == ["e2e"]
    response = client.post("/predict/?model_name=e2e", json=to_records(predict_df))
    assert response.status_code == 200, response.text
    api_predictions = PredictResponse.model_validate(response.json())

    # The CLI and the API give the same predictions for the same model.
    assert [p.id for p in api_predictions.predictions] == list(cli_predictions["id"])
    assert [p.prediction for p in api_predictions.predictions] == list(
        cli_predictions["prediction"]
    )


def test_end_to_end_api(dataset, model_dir):
    """Train and predict only through the API, streaming the result as CSV."""
    train_df, holdout_df = dataset
    predict_df = to_predict_df(holdout_df)
    client = TestClient(api)

    response = client.post("/train/?model_name=e2e", json=to_records(train_df))
    assert response.status_code == 200, response.text
    trained = TrainResponse.model_validate(response.json())
    assert trained.n_samples == len(train_df)
    assert set(trained.classes) == set(train_df["OutcomeType"])
    assert (model_dir / "e2e.joblib").exists()

    response = client.post(
        "/predict/?model_name=e2e&format=csv", json=to_records(predict_df)
    )
    assert response.status_code == 200, response.text
    predictions = pd.read_csv(io.StringIO(response.text), dtype={"id": str})
    assert list(predictions["id"]) == list(predict_df["ID"].astype(str))
    assert set(predictions["prediction"]) <= set(trained.classes)


def test_end_to_end_subset_performance(tmp_path, model_dir):
    """On real data, the served predictions are accurate and fast enough."""
    df = pd.read_csv(DATA_DIR / "train.csv").sample(2500, random_state=0)
    train_df, holdout_df = df.iloc[:-500], df.iloc[-500:]
    train_path = tmp_path / "train.csv"
    train_df.to_csv(train_path, index=False)
    run_cli("train", train_path, model_dir / "e2e.joblib")

    client = TestClient(api)
    start = time.perf_counter()
    response = client.post(
        "/predict/?model_name=e2e", json=to_records(to_predict_df(holdout_df))
    )
    seconds = time.perf_counter() - start
    assert response.status_code == 200, response.text

    y_pred = [p["prediction"] for p in response.json()["predictions"]]
    assert accuracy_score(holdout_df["OutcomeType"], y_pred) >= MIN_SUBSET_ACCURACY
    assert seconds <= MAX_API_PREDICT_SECONDS
