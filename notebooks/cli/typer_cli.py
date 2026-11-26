"""CLI using Typer."""

from typing import Annotated

import typer

app = typer.Typer(
    name="CLI using Typer"
)

@app.command()
def say_hi(name):
    """Say hello to the user."""
    typer.echo(f"Hello: {name}")

@app.command()
def goodbye():
    """Say goodbye to the user."""
    typer.echo("Goodbye!")

@app.command()
def greet(name):
    """Greet the user."""
    typer.echo(f"Greetings {name}!")

@app.command()
def welcome(
    name: Annotated[str, typer.Argument(help="Name of the person to welcome")],
    greeting: Annotated[str, typer.Option(help="Custom greeting text")] = "Hello",
    enthusiastic: Annotated[
        bool, typer.Option("--enthusiastic", help="Add exclamation marks")
    ] = False,
):
    """Welcome the user with an optional custom greeting."""
    message = f"{greeting} {name}"
    if enthusiastic:
        message += "!!!"
    typer.echo(message)

if __name__ == "__main__":
    app()
