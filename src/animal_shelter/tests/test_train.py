import joblib
import numpy as np
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from animal_shelter.model import train


def test_build_transformer():
    transformer = train.build_transformer(["sex"], ["days_upon_outcome"])

    assert isinstance(transformer, ColumnTransformer)
    columns = {name: cols for name, _, cols in transformer.transformers}
    assert columns == {"numeric": ["days_upon_outcome"], "categorical": ["sex"]}


def test_build_transformer_output_shape(features_df):
    transformer = train.build_transformer(["sex"], ["days_upon_outcome"])
    result = transformer.fit_transform(features_df)

    # 1 numeric column + 3 sex categories with the first one dropped.
    assert result.shape == (len(features_df), 1 + 2)


def test_build_transformer_imputes_missing_numeric(features_df):
    transformer = train.build_transformer([], ["days_upon_outcome"])
    result = transformer.fit_transform(features_df)

    assert features_df["days_upon_outcome"].isna().any()
    assert not np.isnan(result).any()


def test_build_model():
    clf_model = train.build_model()

    assert isinstance(clf_model, Pipeline)
    assert list(clf_model.named_steps) == ["transformer", "model"]
    assert isinstance(clf_model.named_steps["model"], RandomForestClassifier)


def test_split_features_target(features_df):
    x, y = train.split_features_target(
        features_df, cat_features=["sex"], num_features=["days_upon_outcome"]
    )

    assert_frame_equal(x, features_df[["sex", "days_upon_outcome"]])
    assert_series_equal(y, features_df["outcome_type"])


def test_split_features_target_defaults(features_df):
    x, y = train.split_features_target(features_df)

    assert list(x.columns) == train.CAT_FEATURES + train.NUM_FEATURES
    assert y.name == train.TARGET


def test_split_features_target_missing_column_raises(features_df):
    with pytest.raises(KeyError):
        train.split_features_target(features_df.drop(columns="sex"))


def test_train_model(fitted_model, features_df):
    x, _ = train.split_features_target(features_df)
    predictions = fitted_model.predict(x)

    assert len(predictions) == len(features_df)
    assert set(predictions) <= set(features_df["outcome_type"])
    assert set(fitted_model.classes_) == set(features_df["outcome_type"])


def test_save_model(fitted_model, features_df, tmp_path):
    path = tmp_path / "model.joblib"
    train.save_model(fitted_model, path)

    assert path.exists()
    loaded = joblib.load(path)
    x, _ = train.split_features_target(features_df)
    assert (loaded.predict(x) == fitted_model.predict(x)).all()


def test_train(raw_data_path, tmp_path):
    model_path = tmp_path / "model.joblib"
    clf_model = train.train(data_path=raw_data_path, model_path=model_path)

    assert isinstance(clf_model, Pipeline)
    assert model_path.exists()
    assert set(clf_model.classes_) == {"Adoption", "Transfer"}
