#!/usr/bin/bash
rm -rf logs

torch-model-archiver --model-name resnet18 \
--version 1.0 \
--serialized-file resnet18.pth \
--extra-files ./code/FVHandler.py \
--handler ./code/serve_handle.py  \
--export-path model_store -f

# local serve
#torchserve --start --foreground --model-store model_store/ --models resnet18=resnet18.mar --ts-config config.properties

# docker cpu
#docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071  pytorch/torchserve:latest-cpu 

# docker gpu
#docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar -v ${PWD}/model_store:/home/model-server/model-store  -v ${PWD}/config.properties:/home/model-server/config.properties  pytorch/torchserve:latest-gpu nvidia-smi
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar -v ${PWD}/model_store:/home/model-server/model-store  -v ${PWD}/config.properties:/home/model-server/config.properties  pytorch/torchserve:latest-gpu torchserve --start --model-store model-store --models resnet18=resnet18.mar --ts-config config.properties
