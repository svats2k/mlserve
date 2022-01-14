import os
import io
import zlib
import time
#from rich.logging import RichHandler
from functools import wraps
from io import BytesIO
import json
import logging
import numpy as np
from PIL import Image
from typing import Callable, List, Tuple

import torch
from torch import nn
import torchvision.transforms as transforms
from ts.torch_handler.base_handler import BaseHandler

from mlserve.common.misc import uncompress_nparr, compress_nparr, stopwatch

# Setting logger
logger = logging.getLogger(__name__)
#shell_handler = RichHandler()
#
##logger.setLevel(logging.DEBUG)
##shell_handler.setLevel(logging.DEBUG)
#
#logger.setLevel(logging.ERROR)
#shell_handler.setLevel(logging.ERROR)
#
## the formatter determines how the logger looks like
#FMT_SHELL = "%(message)s"
#FMT_FILE = """%(levelname)s %(asctime)s [%(filename)s
#    %(funcName)s %(lineno)d] %(message)s"""
#
#shell_formatter = logging.Formatter(FMT_SHELL)
#shell_handler.setFormatter(shell_formatter)
#logger.addHandler(shell_handler)

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
        logger.info(f"properties_dict: {properties}")
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

        logger.info("logging model here ...")
        logger.info(dir(self.model))
        logger.info("-------------------------")

        self.extraction_layer = self.model._modules.get('avgpool')
        self.model = self.model.to(self.device)
        
        self.initialized = True
        
    def save_output_hook(self, layer_id : str) -> Callable:
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self._activation[layer_id] = output.detach()
        return hook

    def _get_feature_vectors(self) -> np.ndarray:
        return self._activation['fvec'].squeeze().cpu().numpy()

class FVHandlerBatch(FVHandler):
    """This handler takes in images in the form of a list and returns the feature vector

    Args:
        BaseHandler ([type]): [description]
    """
    
    def __init__(self):
        super().__init__()

    @stopwatch
    def preprocess(self, data) -> np.ndarray:
        logger.info(f"incoming data type: {type(data)}, incoming length: {len(data)}")
        images_batch = []
        for data_batch in data:
            compressed_byte_string = data[0].get('data') or data[0].get('body')
            imgs_np = uncompress_nparr(compressed_byte_string)
            logger.debug(f"In preprocess data_batch shape {imgs_np.shape}")
            images = torch.stack([self.transform(img_np) for img_np in imgs_np])
            images = images.to(device=self.device, dtype=torch.float32)
            images_batch.append(images)

        return images_batch

    @stopwatch
    def inference(self, image_batches: List[torch.Tensor], *args, **kwargs) -> np.ndarray:
        
        feature_vectors_list = []
        
        for image_batch in image_batches:
            logger.debug(f"In inference image batch shape: {image_batch.shape}")
            h = self.extraction_layer.register_forward_hook(self.save_output_hook('fvec'))
            with torch.no_grad():
                h_x = self.model(image_batch)
            h.remove()
            
            feature_vectors_list.append(self._get_feature_vectors())

        return feature_vectors_list

    @stopwatch
    def postprocess(self, data: List[np.ndarray]) -> List[str]:
        
        comp_byte_string_list = []
        
        logger.info(f"Number of output batches: {len(data)}")
        
        for fv_batch in data:
            compressed_byte_string, _, _ = compress_nparr(fv_batch)
            logger.debug(f"In post process feature vector shape: {fv_batch.shape}")
            comp_byte_string_list.append(compressed_byte_string)

        return comp_byte_string_list

class FVHandlerSingle(BaseHandler):
    """This handler takes in images in the form of a list and returns the feature vector

    Args:
        BaseHandler ([type]): [description]
    """
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
        
    @stopwatch
    def preprocess(self, data) -> np.ndarray:
        logger.info(f"data type: {type(data)}, length: {len(data)}")
        compressed_byte_string = data[0].get('data') or data[0].get('body')
        imgs_np = uncompress_nparr(compressed_byte_string)
        #imgs_np = np.load(BytesIO(bytes_info[0]['body']), allow_pickle=True)
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
