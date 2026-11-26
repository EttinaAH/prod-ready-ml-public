"""Request and response schemas for the API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AnimalRecord(BaseModel):
    """One animal, using the same column names as `data/test.csv`."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, alias="ID")
    name: str | None = Field(default=None, alias="Name")
    date_time: datetime = Field(alias="DateTime")
    animal_type: str = Field(alias="AnimalType")
    sex_upon_outcome: str | None = Field(default=None, alias="SexuponOutcome")
    age_upon_outcome: str | None = Field(default=None, alias="AgeuponOutcome")
    breed: str = Field(alias="Breed")
    color: str | None = Field(default=None, alias="Color")


class LabeledAnimalRecord(AnimalRecord):
    """One animal with a known outcome, as in `data/train.csv`."""

    outcome_type: str = Field(alias="OutcomeType")
    outcome_subtype: str | None = Field(default=None, alias="OutcomeSubtype")


class TrainResponse(BaseModel):
    """Summary of a training run."""

    model_name: str
    n_samples: int
    classes: list[str]
    train_accuracy: float
    duration_seconds: float


class Prediction(BaseModel):
    """Prediction for one animal."""

    id: str
    prediction: str


class PredictResponse(BaseModel):
    """Predictions for a batch of animals."""

    model_name: str
    predictions: list[Prediction]
