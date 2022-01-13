# PyTorch model serving experiments

As a precursor to writing a blog, I would like to document the development of 
various deployments options for a Pytorch model which generates a feature vector 
based on a resnet-18 model.

The following options are being experimented on:
The goal of this repo is to provide an example of deploying an image feature vector using resnet-18 on:
- local CPU using torchserve
- local GPU using torchserve
- AWS DLC endpoint
- Accelerating the inference using Sagemaker Inference

Following the deployment of a single model we will document ways to add multiple 
models and scale them.
