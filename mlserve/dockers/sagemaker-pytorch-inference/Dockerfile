# Creating this image to overwrite the sagemaker_inference library version

# Base SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0.cpu-py38

RUN pip uninstall sagemaker_inference
RUN pip install -U sagemaker_inference
