# Local torchserve deployments

#### Installing dependences
In addition to installing mlserve related requirements, you will want to start 
with one of the following files from https://github.com/pytorch/serve/tree/master/ts_scripts 

1.  Installing for torchserve CPU
- Installing dependencies
```
cd mlserve
mdir tools
git clone https://github.com/pytorch/serve.git torchserve_local
cd torchserve_local
python ./ts_scripts/install_dependencies.py
```
- Install torchserve, torch-model-archiver and torch-workflow-archiver
```
pip install torchserve torch-model-archiver torch-workflow-archiver
```

Please take a look at torchserve documentation [here](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model) for 
additional information
#### Creating a model file from torchvision repo


#### Creating a .mar file


#### Serving the .mar file using torchserve