import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from animal_shelter.model import predict, train


def test_load_model(fitted_model, features_df, tmp_path):
    path = tmp_path / "model.joblib"
    train.save_model(fitted_model, path)
    loaded = predict.load_model(path)

    x = predict.select_features(features_df)
    np.testing.assert_array_equal(loaded.predict(x), fitted_model.predict(x))


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_model(tmp_path / "does_not_exist.joblib")


def test_select_features(features_df):
    result = predict.select_features(
        features_df, cat_features=["sex"], num_features=["days_upon_outcome"]
    )
    assert_frame_equal(result, features_df[["sex", "days_upon_outcome"]])


def test_select_features_drops_target(features_df):
    result = predict.select_features(features_df)

    assert list(result.columns) == train.CAT_FEATURES + train.NUM_FEATURES
    assert train.TARGET not in result.columns


def test_predict_model(fitted_model, features_df):
    # The target column is present but should be ignored.
    predictions = predict.predict_model(fitted_model, features_df)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(features_df),)
    assert set(predictions) <= set(features_df["outcome_type"])


def test_predict(raw_data_path, tmp_path):
    model_path = tmp_path / "model.joblib"
    train.train(data_path=raw_data_path, model_path=model_path)

    predictions = predict.predict(data_path=raw_data_path, model_path=model_path)

    assert predictions.shape == (4,)
    assert set(predictions) <= {"Adoption", "Transfer"}
