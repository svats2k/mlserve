import sys
import time
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

import asyncio
import aiohttp

from decord import VideoLoader, cpu

from typing import Optional

from sagemaker.deserializers import BytesDeserializer

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

@stopwatch
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


async def do_post(session, url, image):
    async with session.post(url, data=image) as response:
        return await response.read()

async def make_predictions(data_stack, model_url):
    async with aiohttp.ClientSession() as session:
        post_tasks = []
        
        # prepare the coroutines that post
        for img in data_stack:
            post_tasks.append(do_post(session, model_url, img))

        # now execute them all at once
        responses = await asyncio.gather(*post_tasks)
        return responses

def get_batch_predictions(data_batch, model_url):
    logger.info(f"Received data batch: {len(data_batch)} and type {type(data_batch)}")
    loop = asyncio.get_event_loop()
    predictions = loop.run_until_complete(make_predictions(data_batch, model_url))
    return predictions

@stopwatch
def compute_fvecs_async(
    vid_batches: VideoLoader,
    pred_url:str="http://localhost:8080/predictions/resnet18"
) -> np.ndarray:

    print(json.loads(requests.get('http://localhost:8081/models').content))

    num_batches = len(vid_batches)
    
    num_frms = 0
    logger.info(f"Num. batches : {num_batches}")
    pbar = tqdm(vid_batches, total=num_batches, position=0, desc="Byte Array gen", leave=True)
    bytearray_list = []
    for vid_batch, _ in pbar:
        byt_str, _, _ = compress_nparr(vid_batch.asnumpy())
        bytearray_list.append(byt_str)
        num_frms += vid_batch.asnumpy().shape[0]
        
    #logger.info(f"Divided the frms into {num_batches} batches of size ~ {round(sys.getsizeof(byarrs[0])/1024**2,2)} MB")

    pred_start = time.perf_counter()
    # Using threadpool
    #with ThreadPoolExecutor(max_workers=4) as executor:
    #    batch_fvecs_list:List[requests.Response] = list(executor.map(lambda byarr: requests.post(pred_url, data=byarr), bytearray_list))
    #fvecs_list:List[np.ndarray] = [uncompress_nparr(resp_ele.content) for resp_ele in batch_fvecs_list]

    # Using asyncio
    batch_fvecs_list = get_batch_predictions(data_batch=bytearray_list, model_url=pred_url)
    logger.info(f"return type : {type(batch_fvecs_list[0])}")
    fvecs_list:List[np.ndarray] = [uncompress_nparr(resp_ele) for resp_ele in batch_fvecs_list]
    logger.info(fvecs_list[0].shape)

    fvecs_array = np.concatenate(fvecs_list)

    elapsed_time = time.perf_counter() - pred_start
    logger.info(f"Time taken for prediction: {elapsed_time:.2f}")
    
    if num_frms != fvecs_array.shape[0]:
        logger.error(f"Input frames {num_frms} not returned by server {fvecs_array.shape}")
        raise typer.Exit()

    logger.info(f"fvec dimensions {fvecs_array.shape} and num frames given {num_frms}")
    
    return fvecs_array

def main(
    video_loc:Path=typer.Option(..., "-i", help="Input Video location", exists=True),
    batch_size:int=typer.Option(50, "-n", help="Number of batches"),
    pred_url:str=typer.Option('http://localhost:8080/predictions/resnet18'),
    num_frames:Optional[int]=typer.Option(None, "-nf", help="number of frames")
):
    vb = get_video_batch_object(video_loc=video_loc.__str__(), batch_size=batch_size)
    #compute_fvecs(vb, pred_url=pred_url)
    compute_fvecs_async(vb, pred_url=pred_url)
    
if __name__ == '__main__':
    typer.run(main)