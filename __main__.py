import json
import typer
from pathlib import Path

from typing import Optional, Dict, List, Sequence

app = typer.Typer(name="Serving ML Models", add_completion=False)


@app.command()
def test_local(
    gpu:bool=typer.Option(False,"-gpu")
):
    return NotImplementedError


@app.command()
def test_local_tsrv():
    return NotImplementedError


@app.command()
def test_aws_dlc():
    return NotImplementedError


@app.command()
def test_aws_sagemaker():
    return NotImplementedError

if __name__ == '__main__':
    app()

