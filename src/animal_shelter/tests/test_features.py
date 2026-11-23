import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from animal_shelter import features


def test_add_features():
    df = pd.DataFrame(
        {
            "animal_type": ["Dog", "Cat"],
            "name": ["Ivo", "Unknown"],
            "sex_upon_outcome": ["Neutered Male", "Intact Female"],
            "breed": ["Pit Bull Mix", "Domestic Shorthair Mix"],
            "age_upon_outcome": ["2 years", "3 weeks"],
        }
    )
    result = features.add_features(df)

    assert_series_equal(result["is_dog"], pd.Series([True, False], name="is_dog"))
    assert_series_equal(result["has_name"], pd.Series([True, False], name="has_name"))
    assert_series_equal(result["sex"], pd.Series(["male", "female"], name="sex"))
    assert_series_equal(
        result["neutered"], pd.Series(["fixed", "intact"], name="neutered")
    )
    assert_series_equal(
        result["hair_type"], pd.Series(["unknown", "shorthair"], name="hair_type")
    )
    assert_series_equal(
        result["days_upon_outcome"],
        pd.Series([730.0, 21.0], name="days_upon_outcome"),
    )
    # The input DataFrame should not be modified.
    assert "is_dog" not in df.columns


def test_check_is_dog():
    s = pd.Series(["Dog", "cat", "DOG"])
    result = features._check_is_dog(s)
    expected = pd.Series([True, False, True])
    assert_series_equal(result, expected)


def test_check_is_dog_raises_on_other_animals():
    s = pd.Series(["Dog", "Bird"])
    with pytest.raises(RuntimeError):
        features._check_is_dog(s)


def test_check_has_name():
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features._check_has_name(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)


@pytest.fixture()
def list_of_sex():
    return pd.Series(["Neutered Male", "Spayed Female", "Intact Male", "Unknown"])


def test_get_sex(list_of_sex):
    result = features._get_sex(list_of_sex)
    expected = pd.Series(["male", "female", "male", "unknown"])
    assert_series_equal(result, expected)


def test_get_neutered(list_of_sex):
    result = features._get_neutered(list_of_sex)
    expected = pd.Series(["fixed", "fixed", "intact", "unknown"])
    assert_series_equal(result, expected)


def test_get_hair_type():
    s = pd.Series(
        [
            "Domestic Shorthair Mix",
            "Domestic Medium Hair",
            "Domestic Longhair Mix",
            "Labrador Retriever Mix",
        ]
    )
    result = features._get_hair_type(s)
    expected = pd.Series(["shorthair", "medium hair", "longhair", "unknown"])
    assert_series_equal(result, expected)


def test_compute_days_upon_outcome():
    s = pd.Series(
        ["1 year", "2 years", "1 month", "2 weeks", "3 days", "1 day", "Unknown"]
    )
    result = features._compute_days_upon_outcome(s)
    expected = pd.Series([365.0, 730.0, 30.0, 14.0, 3.0, 1.0, np.nan])
    assert_series_equal(result, expected)
