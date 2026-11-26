import pandas as pd
import pytest

from animal_shelter import setup_logger


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset the logger after each test.

    The CLI calls `setup_logger`, which attaches a handler to the stderr of
    `CliRunner`. That stream is closed after the command, so later log
    messages would fail with "I/O operation on closed file".
    """
    yield
    setup_logger()


TRAIN_RECORDS = [
    {
        "AnimalID": "A1",
        "Name": "Ivo",
        "DateTime": "2014-02-12 18:22:00",
        "OutcomeType": "Adoption",
        "OutcomeSubtype": None,
        "AnimalType": "Dog",
        "SexuponOutcome": "Neutered Male",
        "AgeuponOutcome": "2 years",
        "Breed": "Pit Bull Mix",
        "Color": "Brown/White",
    },
    {
        "AnimalID": "A2",
        "Name": None,
        "DateTime": "2013-10-13 12:44:00",
        "OutcomeType": "Transfer",
        "OutcomeSubtype": "Partner",
        "AnimalType": "Cat",
        "SexuponOutcome": "Intact Female",
        "AgeuponOutcome": "3 weeks",
        "Breed": "Domestic Shorthair Mix",
        "Color": "Cream Tabby",
    },
    {
        "AnimalID": "A3",
        "Name": "Henk",
        "DateTime": "2015-01-31 12:28:00",
        "OutcomeType": "Adoption",
        "OutcomeSubtype": None,
        "AnimalType": "Dog",
        "SexuponOutcome": "Spayed Female",
        "AgeuponOutcome": "1 year",
        "Breed": "Labrador Retriever Mix",
        "Color": "Black",
    },
    {
        "AnimalID": "A4",
        "Name": "Bella",
        "DateTime": "2014-07-11 19:09:00",
        "OutcomeType": "Transfer",
        "OutcomeSubtype": "Partner",
        "AnimalType": "Cat",
        "SexuponOutcome": "Intact Male",
        "AgeuponOutcome": "1 month",
        "Breed": "Domestic Shorthair Mix",
        "Color": "Orange",
    },
]

_LABEL_COLUMNS = ("AnimalID", "OutcomeType", "OutcomeSubtype")


@pytest.fixture()
def train_records():
    """Return records in the raw format of data/train.csv."""
    return [dict(r) for r in TRAIN_RECORDS]


@pytest.fixture()
def test_records():
    """Return records in the raw format of data/test.csv."""
    return [
        {"ID": str(i), **{k: v for k, v in r.items() if k not in _LABEL_COLUMNS}}
        for i, r in enumerate(TRAIN_RECORDS, start=1)
    ]


@pytest.fixture()
def train_csv(tmp_path, train_records):
    path = tmp_path / "train.csv"
    pd.DataFrame(train_records).to_csv(path, index=False)
    return path


@pytest.fixture()
def test_csv(tmp_path, test_records):
    path = tmp_path / "test.csv"
    pd.DataFrame(test_records).to_csv(path, index=False)
    return path


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    """Point the app settings to an empty, temporary model directory."""
    path = tmp_path / "models"
    monkeypatch.setenv("SHELTER_MODEL_DIR", str(path))
    return path
