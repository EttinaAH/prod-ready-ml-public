import pytest
from shelter_app import service


def test_model_path_for(tmp_path):
    assert service.model_path_for(tmp_path, "my-model_1") == (
        tmp_path / "my-model_1.joblib"
    )


@pytest.mark.parametrize("name", ["", "../evil", "a/b", "model.joblib"])
def test_model_path_for_rejects_invalid_names(tmp_path, name):
    with pytest.raises(ValueError):
        service.model_path_for(tmp_path, name)


def test_records_to_raw_df(train_records):
    df = service.records_to_raw_df(train_records)

    assert len(df) == len(train_records)
    assert "sex_upon_outcome" in df.columns
    assert not df.isna().any().any()


def test_train_and_save(train_records, tmp_path):
    model_path = tmp_path / "sub" / "model.joblib"
    result = service.train_and_save(
        service.records_to_raw_df(train_records), model_path
    )

    assert model_path.exists()
    assert result.model_name == "model"
    assert result.n_samples == len(train_records)
    assert set(result.classes) == {"Adoption", "Transfer"}
    assert 0 <= result.train_accuracy <= 1


def test_train_and_save_without_target_raises(test_records, tmp_path):
    with pytest.raises(ValueError, match="outcome_type"):
        service.train_and_save(
            service.records_to_raw_df(test_records), tmp_path / "model.joblib"
        )


def test_predict_df(train_records, test_records, tmp_path):
    model_path = tmp_path / "model.joblib"
    service.train_and_save(service.records_to_raw_df(train_records), model_path)

    predictions = service.predict_df(
        service.records_to_raw_df(test_records), model_path
    )

    assert list(predictions.columns) == ["id", "prediction"]
    assert list(predictions["id"]) == ["1", "2", "3", "4"]
    assert set(predictions["prediction"]) <= {"Adoption", "Transfer"}


def test_predict_df_missing_model_raises(test_records, tmp_path):
    with pytest.raises(FileNotFoundError):
        service.predict_df(
            service.records_to_raw_df(test_records), tmp_path / "missing.joblib"
        )


def test_list_models(train_records, tmp_path):
    raw_df = service.records_to_raw_df(train_records)
    service.train_and_save(raw_df, tmp_path / "b.joblib")
    service.train_and_save(raw_df, tmp_path / "a.joblib")

    assert service.list_models(tmp_path) == ["a", "b"]
