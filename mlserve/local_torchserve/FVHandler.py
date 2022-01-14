import os
import io
import zlib
from time import time
from functools import wraps
import numpy as np
from typing import Callable, List, Tuple

import torch
from torch import nn
import torchvision.transforms as transforms
from ts.torch_handler.base_handler import BaseHandler

import logging
logger = logging.getLogger('FVHandler')

def stopwatch(func:Callable) -> Callable:
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()

        logger.info(f"Time taken for {wrapped_func.__name__}'s execution: {round(end_time-start_time, 3)} seconds")

        return result
    return wrapped_func

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

class FVHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        
        self.mapping = None
        self.device = None
        self.initialized = False
        self.metrics = None

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        
        self.transform = transforms.Compose([
            self.to_tensor,
            self.scaler,
            self.normalize
        ])
        
        self._activation = {}
        
    def initialize(self, context):
        
        properties = context.system_properties
        properties['limit_max_image_pixels'] = False
        if torch.cuda.is_available():
            logger.info("Setting device to CUDA")
            self.device = torch.device("cuda:" + str(properties.get("gpu_id")))
        else:
            logger.info("Setting device to CPU")
            self.device = torch.device('cpu')

        model_dir = properties.get('model_dir')
        self.model:nn.Module = torch.load(os.path.join(model_dir, "resnet18.pth"))
        self.model.to(self.device)
        self.model.eval()

        self.extraction_layer = self.model._modules.get('avgpool')
        self.model = self.model.to(self.device)
        
        self.initialized = True
        
    def save_output_hook(self, layer_id : str) -> Callable:
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self._activation[layer_id] = output.detach()
        return hook

    def _get_feature_vectors(self) -> np.ndarray:
        return self._activation['fvec'].squeeze().cpu().numpy()


class FVHandlerSingle(FVHandler):
    """This handler takes in images in the form of a list and returns the feature vector

    Args:
        BaseHandler ([type]): [description]
    """
    def __init__(self):
        super().__init__()

    @stopwatch
    def preprocess(self, data) -> np.ndarray:
        logger.info(f"data type: {type(data)}, length: {len(data)}")
        compressed_byte_string = data[0].get('data') or data[0].get('body')
        imgs_np = uncompress_nparr(compressed_byte_string)
        images = torch.stack([self.transform(img_np) for img_np in imgs_np])
        images = images.to(device=self.device, dtype=torch.float32)
        return images

    @stopwatch
    def inference(self, img_data: torch.Tensor, *args, **kwargs) -> np.ndarray:
        h = self.extraction_layer.register_forward_hook(self.save_output_hook('fvec'))
        with torch.no_grad():
            h_x = self.model(img_data)
        h.remove()

        return self._get_feature_vectors()

    @stopwatch
    def postprocess(self, data: np.ndarray) -> List[str]:
        compressed_byte_string, _, _ = compress_nparr(data)
        logger.info(type(compressed_byte_string))
        
        return [compressed_byte_string]

#class FVHandlerBatch(FVHandler):
#    """This handler takes in images in the form of a list and returns the feature vector
#
#    Args:
#        BaseHandler ([type]): [description]
#    """
#    
#    def __init__(self):
#        super().__init__()
#
#    @stopwatch
#    def preprocess(self, data) -> np.ndarray:
#        logger.info(f"incoming data type: {type(data)}, incoming length: {len(data)}")
#        images_batch = []
#        data_list = data[0].get('data') or data[0].get('body')
#        logger.info(f"Number of batches received: {len(data_list)}")
#        for data_batch in data_list:
#            imgs_np = uncompress_nparr(data_batch)
#            logger.debug(f"In preprocess data_batch shape {imgs_np.shape}")
#            images = torch.stack([self.transform(img_np) for img_np in imgs_np])
#            images = images.to(device=self.device, dtype=torch.float32)
#            images_batch.append(images)
#
#        return images_batch
#
#    @stopwatch
#    def inference(self, image_batches: List[torch.Tensor], *args, **kwargs) -> np.ndarray:
#        
#        feature_vectors_list = []
#        
#        for image_batch in image_batches:
#            logger.debug(f"In inference image batch shape: {image_batch.shape}")
#            h = self.extraction_layer.register_forward_hook(self.save_output_hook('fvec'))
#            with torch.no_grad():
#                h_x = self.model(image_batch)
#            h.remove()
#            
#            feature_vectors_list.append(self._get_feature_vectors())
#
#        return feature_vectors_list
#
#    @stopwatch
#    def postprocess(self, data: List[np.ndarray]) -> List[str]:
#        
#        comp_byte_string_list = []
#        
#        logger.info(f"Number of output batches: {len(data)}")
#        
#        for fv_batch in data:
#            compressed_byte_string, _, _ = compress_nparr(fv_batch)
#            logger.debug(f"In post process feature vector shape: {fv_batch.shape}")
#            comp_byte_string_list.append(compressed_byte_string)
#
#        return comp_byte_string_list