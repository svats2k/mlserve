#!/usr/bin/bash
rm -rf logs

if [ $1 = 'regenmar' ]; then
torch-model-archiver --model-name resnet18 \
--version 1.0 \
--serialized-file resnet18.pth \
--extra-files ./FVHandler.py \
--handler ./serve_handle.py  \
--export-path model_store -f
else
    echo "Not regenrating mar file"
fi

# local serve
#torchserve --start --foreground --model-store model_store/ --models resnet18=resnet18.mar --ts-config config.properties

# docker cpu
#docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071  pytorch/torchserve:latest-cpu 

# docker gpu
#docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar -v ${PWD}/model_store:/home/model-server/model-store  -v ${PWD}/config.properties:/home/model-server/config.properties  pytorch/torchserve:latest-gpu nvidia-smi
#docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar \
# -v ${PWD}/model_store:/home/model-server/model-store  \
# -v ${PWD}/config.properties:/home/model-server/config.properties  \
# pytorch/torchserve:latest-gpu torchserve --start --model-store model-store \
# --models resnet18=resnet18.mar --ts-config config.properties

# aws DLC
docker run --rm -it --name torchserve -v ${PWD}:/tmp/ -p 8080:8080  -p 8081:8081 \
763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38-ubuntu20.04-e3 \
torchserve --start --model-store /tmp/model_store \
--models resnet18=resnet18.mar --ts-config /tmp/config.properties
