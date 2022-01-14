# Serving model on AWS

### Creating a model.tar.gz file

The directory structure needed for Sagemaker is:

```
model.tar.gz/
|-model.pth
|-code/
    |- inference.py
    |- requirements.txt
```