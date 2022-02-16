from __future__ import absolute_import

import os
import logging
from typing import Callable

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
        logger.info('model_fn')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = resnet18(pretrained=True)

        with open(os.path.join(model_dir, 'resnet18.pth'), 'rb') as f:
            model.load_state_dict(torch.load(f))
            
        model.to(device=device)
        model.eval()
        return model
    except Exception as e:
        logger.exception(f"Exception in model fn {e}")
    
def input_fn(input_data, content_type) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np_array = decoder.decode(input_data, content_type)
    
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
        
    transform = transforms.Compose([
        to_tensor,
        scaler,
        normalize
    ])

    images = torch.stack([transform(img_np) for img_np in np_array])
    torch_arr = images.to(device=device, dtype=torch.float32)

    return torch_arr

def predict_fn(input_data: torch.Tensor, model: nn.Module):

    activation = {}

    def save_output_hook(self, layer_id : str) -> Callable:
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor):
            activation[layer_id] = output.detach()
        return hook

    extraction_layer = model._modules.get('avgpool')

    h = extraction_layer.register_forward_hook(save_output_hook('fvec'))
    with torch.no_grad():
        h_x = model(input_data)

    h.remove()
    
    return activation['fvec']

def output_fn(prediction, output_response_type):

    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().cpu().numpy().tolist()

    for content_type in utils.parse_accept(output_response_type):
        if content_type in encoder.SUPPORTED_CONTENT_TYPES:
            encoded_prediction = encoder.encode(prediction, content_type)
            if content_type == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")
            return encoded_prediction

    raise errors.UnsupportedFormatError(output_response_type)