# Imports
import sys
import json
from pathlib import Path
import numpy as np
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from mlserve.common.logger import logger
from mlserve.local_serve.load_model import Img2Vec
from mlserve.common.misc import stopwatch

app = FastAPI()

# Load the model data
model_loc:Path = Path("../models/resnet18.pth")
if model_loc.exists():
    logger.info(f"Loading model from {model_loc.__str__()}")
    img2vec = Img2Vec(cuda=True, resnet_model18_path=model_loc.__str__())
else:
    logger.error(f"Missing model file: {model_loc.resolve().__str__()}")
    sys.exit()
class Request(BaseModel):
    lvecs: List[Any]
    
class Response(BaseModel):
    fvecs: List[List[float]]
    
@app.get("/say_hi")
async def say_hi():
    return ("You have reached the FASTAPI server for getting feature of images using resnet-18 model")

@stopwatch
@app.post('/get_vecs', response_model=Response)
async def get_vecs(data: Request) -> Dict:
    frms = np.array(json.loads(data.json())['lvecs'])
    logger.info(f"Input data shape: {frms.shape}")

    fvecs = img2vec.get_vecs(frms)
    logger.info(f"Feature vector shape: {fvecs.shape}")
    
    return {'fvecs': fvecs.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)