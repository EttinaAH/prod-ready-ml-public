"""Tests that call functions many times with randomly generated input.

Each test is repeated for a list of seeds, so a failure can always be
reproduced by re-running the test with the seed shown in its name.
"""

import numpy as np
import pandas as pd
import pytest

from animal_shelter.data import convert_camel_case, load_data
from animal_shelter.features import add_features
from animal_shelter.model import predict, train

SEEDS = range(25)

ANIMAL_TYPES = ["Dog", "Cat"]
SEX_UPON_OUTCOME = [
    "Neutered Male",
    "Spayed Female",
    "Intact Male",
    "Intact Female",
    "Unknown",
]
AGE_UNITS = {
    "day": 1,
    "days": 1,
    "week": 7,
    "weeks": 7,
    "month": 30,
    "months": 30,
    "year": 365,
    "years": 365,
}
BREEDS = [
    "Domestic Shorthair Mix",
    "Domestic Medium Hair Mix",
    "Domestic Longhair",
    "Labrador Retriever Mix",
    "Pit Bull/Boxer",
]
OUTCOMES = ["Adoption", "Transfer", "Return_to_owner", "Euthanasia", "Died"]


def make_raw_df(n_rows: int, seed: int) -> pd.DataFrame:
    """Generate random data in the raw format of data/train.csv."""
    rng = np.random.default_rng(seed)
    letters = list("abcdefghijklmnopqrstuvwxyz")

    def random_name() -> str | None:
        if rng.random() < 0.3:
            return None
        return "".join(rng.choice(letters, size=rng.integers(2, 10))).title()

    def random_age() -> str | None:
        if rng.random() < 0.05:
            return None
        return f"{rng.integers(0, 21)} {rng.choice(list(AGE_UNITS))}"

    return pd.DataFrame(
        {
            "AnimalID": [f"A{i}" for i in range(n_rows)],
            "Name": [random_name() for _ in range(n_rows)],
            "DateTime": pd.Timestamp("2013-10-01")
            + pd.to_timedelta(rng.integers(0, 3 * 365 * 24 * 60, n_rows), unit="min"),
            "OutcomeType": rng.choice(OUTCOMES, n_rows),
            "OutcomeSubtype": rng.choice(["Partner", "Foster", None], n_rows),
            "AnimalType": rng.choice(ANIMAL_TYPES, n_rows),
            "SexuponOutcome": rng.choice(SEX_UPON_OUTCOME, n_rows),
            "AgeuponOutcome": [random_age() for _ in range(n_rows)],
            "Breed": rng.choice(BREEDS, n_rows),
            "Color": rng.choice(["Black", "White", "Brown Tabby"], n_rows),
        }
    )


def load_random_data(tmp_path, n_rows: int, seed: int) -> pd.DataFrame:
    """Write random raw data to a CSV and load it with `load_data`."""
    path = tmp_path / f"random_{seed}.csv"
    make_raw_df(n_rows, seed).to_csv(path, index=False)
    return load_data(path)


@pytest.mark.parametrize("seed", SEEDS)
def test_convert_camel_case_random_words(seed):
    rng = np.random.default_rng(seed)
    words = [
        "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), rng.integers(2, 8)))
        for _ in range(rng.integers(1, 5))
    ]
    camel_case = "".join(word.capitalize() for word in words)

    assert convert_camel_case(camel_case) == "_".join(words)


@pytest.mark.parametrize("seed", SEEDS)
def test_load_data_random_input(tmp_path, seed):
    n_rows = int(np.random.default_rng(seed).integers(1, 200))
    df = load_random_data(tmp_path, n_rows, seed)

    assert len(df) == n_rows
    assert not df.isna().any().any()
    assert all(col == col.lower() for col in df.columns)


@pytest.mark.parametrize("seed", SEEDS)
def test_add_features_random_input(tmp_path, seed):
    n_rows = int(np.random.default_rng(seed).integers(1, 200))
    df = load_random_data(tmp_path, n_rows, seed)
    original = df.copy()

    result = add_features(df)

    # The input is not modified and all original columns are kept.
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(result[original.columns], original)

    assert (result["is_dog"] == (df["animal_type"] == "Dog")).all()
    assert (result["has_name"] == (df["name"] != "Unknown")).all()

    expected_sex = np.select(
        [
            df["sex_upon_outcome"].str.endswith("Female"),
            df["sex_upon_outcome"].str.endswith("Male"),
        ],
        ["female", "male"],
        default="unknown",
    )
    assert (result["sex"] == expected_sex).all()
    assert set(result["neutered"]) <= {"fixed", "intact", "unknown"}
    assert set(result["hair_type"]) <= {
        "shorthair",
        "medium hair",
        "longhair",
        "unknown",
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_days_upon_outcome_random_input(tmp_path, seed):
    df = add_features(load_random_data(tmp_path, 100, seed))

    is_known = df["age_upon_outcome"] != "Unknown"
    split_age = df.loc[is_known, "age_upon_outcome"].str.split()
    expected = split_age.str[0].astype(float) * split_age.str[1].map(AGE_UNITS)

    pd.testing.assert_series_equal(
        df.loc[is_known, "days_upon_outcome"], expected, check_names=False
    )
    assert df.loc[~is_known, "days_upon_outcome"].isna().all()
    assert (df.loc[is_known, "days_upon_outcome"] >= 0).all()


@pytest.mark.parametrize("seed", range(5))
def test_train_and_predict_random_input(tmp_path, seed):
    df = add_features(load_random_data(tmp_path, 200, seed))

    clf_model = train.train_model(df)
    predictions = predict.predict_model(clf_model, df)

    assert predictions.shape == (len(df),)
    assert set(predictions) <= set(df[train.TARGET])
    assert set(clf_model.classes_) == set(df[train.TARGET])


@pytest.mark.parametrize("seed", range(5))
def test_train_is_reproducible(tmp_path, seed):
    df = add_features(load_random_data(tmp_path, 200, seed))

    first = predict.predict_model(train.train_model(df), df)
    second = predict.predict_model(train.train_model(df), df)

    np.testing.assert_array_equal(first, second)
