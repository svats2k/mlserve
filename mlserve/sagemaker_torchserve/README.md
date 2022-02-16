# Serving model on AWS

### Installations
1. pip install 'sagemaker[local]' --upgrade


### Creating a model.tar.gz file

The directory structure needed for Sagemaker is:

```
model.tar.gz/
|-model.pth
|-code/
    |- inference.py
    |- requirements.txt
```