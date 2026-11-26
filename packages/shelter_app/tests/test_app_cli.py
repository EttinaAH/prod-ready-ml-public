import pandas as pd
from shelter_app.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_train(train_csv, tmp_path):
    model_path = tmp_path / "models" / "model.joblib"
    result = runner.invoke(app, ["train", str(train_csv), str(model_path)])

    assert result.exit_code == 0, result.output
    assert model_path.exists()
    assert "Trained model on 4 rows" in result.output


def test_train_missing_input_fails(tmp_path):
    result = runner.invoke(
        app, ["train", str(tmp_path / "missing.csv"), str(tmp_path / "m.joblib")]
    )

    assert result.exit_code != 0


def test_train_without_outcome_fails(test_csv, tmp_path):
    model_path = tmp_path / "model.joblib"
    result = runner.invoke(app, ["train", str(test_csv), str(model_path)])

    assert result.exit_code == 1
    assert not model_path.exists()


def test_predict(train_csv, test_csv, tmp_path):
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "out" / "predictions.csv"
    runner.invoke(app, ["train", str(train_csv), str(model_path)])

    result = runner.invoke(
        app, ["predict", str(test_csv), str(model_path), str(output_path)]
    )

    assert result.exit_code == 0, result.output
    predictions = pd.read_csv(output_path, dtype={"id": str})
    assert list(predictions.columns) == ["id", "prediction"]
    assert list(predictions["id"]) == ["1", "2", "3", "4"]
    assert "Wrote 4 predictions" in result.output


def test_predict_missing_model_fails(test_csv, tmp_path):
    output_path = tmp_path / "predictions.csv"
    result = runner.invoke(
        app,
        ["predict", str(test_csv), str(tmp_path / "missing.joblib"), str(output_path)],
    )

    assert result.exit_code == 1
    assert not output_path.exists()
