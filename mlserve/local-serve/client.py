# Getting predictions from a deployed resnet-18 model
from pathlib import Path
import typer
import decord

from mlserve

app = typer.Typer(name="Deployment tests", add_completion=False)

def load_video(
    vid_loc:Path = typer.Option(..., "-i", help="Video Location")
) -> decord.VideoReader:
    vr = decord.VideoReader(vid_loc)
    return decord.VideoReader

def get_preds(
    vid_loc:Path = typer.Option(..., "-i", help="Video Location"),
    num_frames:int = typer.Option(100, "-n", help="NUmber of frames to process")
) -> None:


if __name__ == '__main__':
    app()
