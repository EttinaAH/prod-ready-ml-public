"""FastAPI app exposing `train` and `predict` endpoints."""

import io
import logging
from enum import StrEnum
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from shelter_app import service
from shelter_app.config import Settings, get_settings
from shelter_app.schemas import (
    AnimalRecord,
    LabeledAnimalRecord,
    PredictResponse,
    TrainResponse,
)

logger = logging.getLogger("animal_shelter.api")

app = FastAPI(title="Animal Shelter API")

SettingsDep = Annotated[Settings, Depends(get_settings)]


class ResponseFormat(StrEnum):
    """Supported formats for the predictions."""

    json = "json"
    csv = "csv"


def _model_path(settings: Settings, model_name: str | None):
    try:
        return service.model_path_for(
            settings.model_dir, model_name or settings.default_model_name
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/health/")
def health() -> dict[str, str]:
    """Check that the API is up."""
    return {"status": "ok"}


@app.get("/models/")
def models(settings: SettingsDep) -> list[str]:
    """List the names of the trained models."""
    return service.list_models(settings.model_dir)


@app.post("/train/")
def train(
    records: list[LabeledAnimalRecord],
    settings: SettingsDep,
    model_name: str | None = None,
) -> TrainResponse:
    """Train a model on the given records and save it under `model_name`."""
    model_path = _model_path(settings, model_name)
    logger.info("Received %d records to train model %s", len(records), model_path)
    if not records:
        raise HTTPException(status_code=422, detail="No training records given")

    raw_df = service.records_to_raw_df([r.model_dump(by_alias=True) for r in records])
    try:
        result = service.train_and_save(raw_df, model_path)
    except (ValueError, RuntimeError) as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=422, detail=str(e)) from e

    return TrainResponse(
        model_name=result.model_name,
        n_samples=result.n_samples,
        classes=result.classes,
        train_accuracy=result.train_accuracy,
        duration_seconds=result.duration_seconds,
    )


@app.post("/predict/", response_model=PredictResponse)
def predict(
    records: list[AnimalRecord],
    settings: SettingsDep,
    model_name: str | None = None,
    response_format: Annotated[ResponseFormat, Query(alias="format")] = (
        ResponseFormat.json
    ),
):
    """Predict outcomes for the given records with the model `model_name`.

    Use `?format=csv` to get the predictions streamed back as a CSV file.
    """
    model_path = _model_path(settings, model_name)
    logger.info("Received %d records to predict with %s", len(records), model_path)
    if not records:
        raise HTTPException(status_code=422, detail="No records given")

    raw_df = service.records_to_raw_df([r.model_dump(by_alias=True) for r in records])
    try:
        predictions = service.predict_df(raw_df, model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, RuntimeError) as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=str(e)) from e

    if response_format == ResponseFormat.csv:
        buffer = io.StringIO()
        predictions.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    return PredictResponse(
        model_name=model_path.stem,
        predictions=predictions.to_dict(orient="records"),
    )
