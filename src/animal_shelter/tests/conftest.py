import pandas as pd
import pytest

from animal_shelter.model import train


@pytest.fixture()
def features_df():
    """Small DataFrame with the columns produced by add_features."""
    return pd.DataFrame(
        {
            "animal_type": ["Dog", "Cat", "Dog", "Cat", "Dog", "Cat"],
            "is_dog": [True, False, True, False, True, False],
            "has_name": [True, False, True, True, False, False],
            "sex": ["male", "female", "female", "male", "unknown", "female"],
            "hair_type": [
                "unknown",
                "shorthair",
                "unknown",
                "longhair",
                "unknown",
                "shorthair",
            ],
            "days_upon_outcome": [730.0, 21.0, 365.0, None, 30.0, 14.0],
            "outcome_type": [
                "Adoption",
                "Transfer",
                "Return_to_owner",
                "Adoption",
                "Transfer",
                "Adoption",
            ],
        }
    )


@pytest.fixture()
def fitted_model(features_df):
    return train.train_model(features_df)


@pytest.fixture()
def raw_data_path(tmp_path):
    """CSV file in the raw format of data/train.csv."""
    df = pd.DataFrame(
        {
            "AnimalID": ["A1", "A2", "A3", "A4"],
            "Name": ["Ivo", None, "Henk", "Bella"],
            "DateTime": [
                "2014-02-12 18:22:00",
                "2013-10-13 12:44:00",
                "2015-01-31 12:28:00",
                "2014-07-11 19:09:00",
            ],
            "OutcomeType": ["Adoption", "Transfer", "Adoption", "Transfer"],
            "OutcomeSubtype": [None, "Partner", None, "Partner"],
            "AnimalType": ["Dog", "Cat", "Dog", "Cat"],
            "SexuponOutcome": [
                "Neutered Male",
                "Intact Female",
                "Spayed Female",
                "Intact Male",
            ],
            "AgeuponOutcome": ["2 years", "3 weeks", "1 year", "1 month"],
            "Breed": [
                "Pit Bull Mix",
                "Domestic Shorthair Mix",
                "Labrador Retriever Mix",
                "Domestic Shorthair Mix",
            ],
            "Color": ["Brown/White", "Cream Tabby", "Black", "Orange"],
        }
    )
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path
