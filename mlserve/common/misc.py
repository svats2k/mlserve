import io
import zlib
from time import time
from functools import wraps
from typing import Callable, Tuple
from pathlib import Path

import numpy as np

import torch
from torchvision import models

from mlserve.common.logger import logger

def stopwatch(func:Callable) -> Callable:
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()

        logger.info(f"Time taken for {wrapped_func.__name__}'s execution: {round(end_time-start_time, 3)} seconds")

        return result
    return wrapped_func


def download_resnet18(save_dir:str='./models/') -> None:
    
    save_dir:Path = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    resnet18 = models.resnet18(pretrained=True)
    
    torch.save(resnet18, save_dir/'resnet18.pth')
    
def compress_nparr(nparr: np.ndarray) -> Tuple[bytes, int, int]:
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    logger.debug(f"Input length: {len(uncompressed)}, compressed length: {len(compressed)}")
    return compressed, len(uncompressed), len(compressed)

def uncompress_nparr(bytestring: bytes) -> np.ndarray:
    return np.load(io.BytesIO(zlib.decompress(bytestring)))
