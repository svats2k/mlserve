from __future__ import absolute_import

import time
import os
import logging
from typing import Callable

from pathlib import Path

from PIL import Image
import numpy as np

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import transforms

from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
    errors,
    utils,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = resnet18(pretrained=True)
        model_path = os.path.join(model_dir, 'model.pth')
        with open(model_path, 'rb') as f:
            logger.info("-I- model loaded from ", model_path)
            model.load_state_dict(torch.load(f))

        model.to(device=device)
        model.eval()
        logger.info("-I- loaded model resnet18")
        return model
    except Exception as e:
        logger.exception(f"Exception in model fn {e}")
    

def predict_fn(input_data: torch.Tensor, model: nn.Module):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    activation = {}

    def save_output_hook(layer_id : str) -> Callable:
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            activation[layer_id] = output.detach()
        return hook

    extraction_layer = model._modules.get('avgpool')

    try:
        np_array = input_data.numpy()
        scaler = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        
        transform = transforms.Compose([
            scaler,
            to_tensor,
            normalize
        ])

        images = torch.stack([transform(Image.fromarray(img_np.astype(np.float32), mode='RGB')) for img_np in np_array])
        torch_arr = images.to(device=device, dtype=torch.float32)
    except Exception as e:
        logger.exception(f"Exception in predict_fn {e}")
        logger.info("-E- unable to convert input to tensor")
        return None

    h = extraction_layer.register_forward_hook(save_output_hook('fvec'))

    with torch.no_grad():
        h_x = model(torch_arr)

    h.remove()
    
    return np.squeeze(activation['fvec'])
