"""Typer CLI exposing `train`, `predict` and `serve` subcommands."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from animal_shelter import setup_logger
from animal_shelter.data import load_data
from shelter_app import service

logger = logging.getLogger("animal_shelter.cli")

app = typer.Typer(name="shelter", help="Train and apply the animal shelter model.")

ExistingFile = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, readable=True)
]


# Always gets called before all subcommands
@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show debug logging.")
    ] = False,
) -> None:
    """Train and apply the animal shelter model."""
    setup_logger(level=logging.DEBUG if verbose else logging.INFO)


@app.command()
def train(input_path: ExistingFile, model_path: Path) -> None:
    """Train a model on the dataset at INPUT_PATH and save it to MODEL_PATH."""
    logger.info("Loading input dataset from %s", input_path)
    raw_df = load_data(input_path)

    try:
        result = service.train_and_save(raw_df, model_path)
    except (ValueError, RuntimeError) as e:
        logger.error("Training failed: %s", e)
        raise typer.Exit(code=1) from e

    typer.echo(
        f"Trained model on {result.n_samples} rows "
        f"(classes: {', '.join(result.classes)}; "
        f"train accuracy: {result.train_accuracy:.3f}) "
        f"in {result.duration_seconds:.2f}s"
    )
    typer.echo(f"Saved model to {result.model_path}")


@app.command()
def predict(input_path: ExistingFile, model_path: Path, output_path: Path) -> None:
    """Apply the model at MODEL_PATH to INPUT_PATH and write a CSV to OUTPUT_PATH."""
    logger.info("Loading input dataset from %s", input_path)
    raw_df = load_data(input_path)

    try:
        predictions = service.predict_df(raw_df, model_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        raise typer.Exit(code=1) from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)

    counts = predictions["prediction"].value_counts()
    typer.echo(f"Wrote {len(predictions)} predictions to {output_path}")
    for outcome, count in counts.items():
        typer.echo(f"  {outcome}: {count}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Start the API (see shelter_app.api) with uvicorn."""
    import uvicorn

    logger.info("Starting API on http://%s:%d", host, port)
    uvicorn.run("shelter_app.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
