import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from shelter_app.api import app


@pytest.fixture()
def client(model_dir):
    return TestClient(app)


@pytest.fixture()
def trained_client(client, train_records):
    response = client.post("/train/?model_name=test", json=train_records)
    assert response.status_code == 200, response.text
    return client


def test_health(client):
    response = client.get("/health/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_train(client, model_dir, train_records):
    response = client.post("/train/?model_name=test", json=train_records)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_name"] == "test"
    assert body["n_samples"] == len(train_records)
    assert set(body["classes"]) == {"Adoption", "Transfer"}
    assert (model_dir / "test.joblib").exists()


def test_train_default_model_name(client, model_dir, train_records):
    response = client.post("/train/", json=train_records)

    assert response.status_code == 200, response.text
    assert (model_dir / "model.joblib").exists()


def test_train_without_outcome_is_rejected(client, test_records):
    response = client.post("/train/", json=test_records)

    assert response.status_code == 422


def test_train_empty_is_rejected(client):
    response = client.post("/train/", json=[])

    assert response.status_code == 422


def test_train_invalid_model_name_is_rejected(client, model_dir, train_records):
    response = client.post("/train/?model_name=../evil", json=train_records)

    assert response.status_code == 422
    assert not model_dir.exists()


def test_models(trained_client):
    response = trained_client.get("/models/")

    assert response.status_code == 200
    assert response.json() == ["test"]


def test_predict(trained_client, test_records):
    response = trained_client.post("/predict/?model_name=test", json=test_records)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_name"] == "test"
    assert [p["id"] for p in body["predictions"]] == ["1", "2", "3", "4"]
    assert {p["prediction"] for p in body["predictions"]} <= {"Adoption", "Transfer"}


def test_predict_csv(trained_client, test_records):
    response = trained_client.post(
        "/predict/?model_name=test&format=csv", json=test_records
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    df = pd.read_csv(io.StringIO(response.text))
    assert list(df.columns) == ["id", "prediction"]
    assert len(df) == len(test_records)


def test_predict_unknown_model_returns_404(client, test_records):
    response = client.post("/predict/?model_name=missing", json=test_records)

    assert response.status_code == 404


def test_predict_missing_required_field_is_rejected(trained_client, test_records):
    del test_records[0]["AnimalType"]
    response = trained_client.post("/predict/?model_name=test", json=test_records)

    assert response.status_code == 422
