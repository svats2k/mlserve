import errno
import os

from time import time
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from typing import Callable, List

from mlserve.common.logger import logger
from mlserve.common.misc import stopwatch

class Img2Vec():

    def __init__(self, cuda:bool=False, resnet_model18_path:str='./models/'):
        
        if not Path(resnet_model18_path).exists():
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                resnet_model18_path
            )
        
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        #Load model
        self.model = torch.load(resnet_model18_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.extraction_layer = self.model._modules.get('avgpool')

        # Tensor transofrmations
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        self._activation = {}

    def save_output_hook(self, layer_id : str) -> Callable:
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self._activation[layer_id] = output.detach()
        return hook

    def _get_feature_vectors(self) -> np.ndarray:
        return self._activation['fvec'].squeeze().cpu().numpy()

    @stopwatch
    def get_vecs(self, imgs_batch:np.ndarray) -> np.ndarray:
        images_list = []
        for idx in range(imgs_batch.shape[0]):
            image = Image.fromarray(imgs_batch[idx].astype(np.uint8))
            image_tensor = self.normalize(self.to_tensor(self.scaler(image)))
            images_list.append(image_tensor)
            
        images_tensor = torch.stack(images_list).to(device=self.device, dtype=torch.float32)
        h = self.extraction_layer.register_forward_hook(self.save_output_hook('fvec'))
        with torch.no_grad():
            h_x = self.model(images_tensor)
        h.remove()

        return  self._get_feature_vectors()
