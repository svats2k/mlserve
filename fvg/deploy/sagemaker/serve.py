import os
import logging
import sys
import json

import numpy as np
from numpy.core.fromnumeric import shape
import torch
from torch import nn
from sagemaker_inference import decoder, content_types, encoder

from feature_vector_generation import predict

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("model")

def model_fn(model_dir:str) -> nn.Module:
    log.info("In model_fn(). DEPLOY_ENV=",os.environ.get("DEPLOY_ENV"))

# From docs:
# Default json deserialization requires request_body contain a single json list.
# https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py#L49
def input_fn(request_body, request_content_type) -> np.ndarray:
    data = json.loads(request_body)
    model_input = np.array(data)
    
    # Check for the dimensions of the numpy array
    if len(model_input.shape) != 4:
        raise ValueError("We need a batch of HxWx3 images")
    if model_input.shape[3] != 3:
        raise ValueError("Images need to have 3 channels RGB")

    return model_input

def predict_fn(input_data: np.ndarray, model: nn.Module) -> np.ndarray:
    prediction = predict.predict(inputs=input_data)
    return prediction

def output_fn(prediction: np.ndarray, content_type) -> json.JSONEncoder:
    res = [{"probabilities": result["probabilities"], "top_n_grams": result["top_n_grams"]} for result in prediction]
    
    return encoder.encode(res, content_type)

