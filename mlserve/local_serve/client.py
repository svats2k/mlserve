# Getting predictions from a deployed resnet-18 model

import json
import requests
import numpy as np
from pathlib import Path
import typer
import decord

from mlserve.common.logger import logger
from mlserve.common.misc import stopwatch

app = typer.Typer(name="Deployment tests", add_completion=False)

def load_video(
    vid_loc:Path = typer.Option(..., "-i", help="Video Location")
) -> decord.VideoReader:
    vr = decord.VideoReader(vid_loc)
    return vr

@app.command()
@stopwatch
def get_preds(
    vid_loc:Path = typer.Option("/home/srivatsas/work/data/sample-mp4-file.mp4", "-i", help="Video Location", exists=True),
    num_frames:int = typer.Option(10, "-n", help="NUmber of frames to process")
) -> None:
    vr = load_video(vid_loc=vid_loc.__str__())
    np_arr:np.ndarray = vr.get_batch(indices=range(num_frames)).asnumpy()

    # Ping the server
    url = 'http://127.00.1:8000/say_hi'
    print(json.loads((requests.get(url=url)).content))
    
    # Get predictions
    logger.info(f"Requesting prediction for {np_arr.shape[0]} frames")
    response = requests.post(
        url='http://127.0.0.1:8000/get_vecs',
        json={'lvecs':np_arr.tolist()}
        #data=json.dumps({'lvecs':np_arr.tolist()})
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        fvecs = np.array(json.loads(response.content)['fvecs'])
        logger.info(f"Obtained feature vectors of shape: {fvecs.shape}")
    else:
        logger.error("Expected response not received")

if __name__ == '__main__':
    app()
