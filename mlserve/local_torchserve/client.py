import sys
import json
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import typer

import msgpack
import msgpack_numpy as m
m.patch()


from decord import VideoLoader, cpu

from typing import Optional

from mlserve.common.logger import logger
from mlserve.common.misc import compress_nparr, uncompress_nparr, stopwatch
 
app = typer.Typer(name="Test local torch serve deployment", add_completion=False)
 
def get_video_batch_object(video_loc:str, batch_size:int=500):

    if not Path(video_loc).exists():
        logger.error(f"Missing file : {video_loc}")
        sys.exit()

    # Resizing images to resnet requirements
    vid_batches = VideoLoader(
        [video_loc],
        ctx=[cpu(0)],
        shape=(batch_size, 224, 224, 3),
        interval=0,
        skip=5,
        shuffle=0
    )
    print('Total batches:', len(vid_batches))
    
    return vid_batches

def compute_fvecs(
    vid_batches: VideoLoader,
    pred_url:str="http://localhost:8080/predictions/resnet18"
) -> np.ndarray:

    print(json.loads(requests.get('http://localhost:8081/models').content))

    num_batches = len(vid_batches)
    
    fvecs_list = []
    num_frms = 0
    pbar = tqdm(vid_batches, total=num_batches, position=0, desc="Byte Array gen", leave=True)
    for vid_batch, _ in pbar:
        logger.debug(f"Adding a batch of shape {vid_batch.shape} to byte array")
        byt_str, _, _ = compress_nparr(vid_batch.asnumpy())
        num_frms += vid_batch.asnumpy().shape[0]
        response = requests.post(url=pred_url, data=byt_str)
        if response.status_code == 200:
            fvecs_list.append(uncompress_nparr(response.content))
        else:
            logger.error('Issue in Server processing the input')
            raise typer.Exit()
        
    #logger.info(f"Divided the frms into {num_batches} batches of size ~ {round(sys.getsizeof(byarrs[0])/1024**2,2)} MB")

    fvecs_array = np.vstack(fvecs_list)
    
    if num_frms != fvecs_array.shape[0]:
        logger.error(f"Input frames {num_frms} not returned by server {fvecs_array.shape}")
        raise typer.Exit()

    logger.info(f"fvec dimensions {fvecs_array.shape} and num frames given {num_frms}")
    
    return fvecs_array

def main(
    video_loc:Path=typer.Option(..., "-i", help="Input Video location", exists=True),
    batch_size:int=typer.Option(50, "-n", help="Number of batches"),
    pred_url:str=typer.Option('http://localhost:8080/predictions/resnet18')
):
    vb = get_video_batch_object(video_loc=video_loc.__str__(), batch_size=batch_size)
    compute_fvecs(vb, pred_url=pred_url)
    
if __name__ == '__main__':
    typer.run(main)