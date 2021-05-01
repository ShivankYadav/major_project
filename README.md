# Major_project
This repository contains contents of major project 2021

## Steps to run
  **Docker Commands**
  1. Build the image from dockerfile using ```docker image build -t proto5 Dockerfile.gpu```
  2. Build the container from the image using ```docker container run -it --network=host --runtime=nvidia --gpus all --name mrcnn_gpu -v $(pwd):/host proto5```
  3. If the container is already created, use the below command to get it up and running:
        1. ```docker container start mrcnn_gpu```
        2. ```docker container exec -it mrcnn_gpu /bin/bash```
        
  **Commands to run inside the container**
  (Everytime commands)
  1. ```cd /host```
  
  (Only to be run If created the container for the first time)
  1. ```pip3 install -r requirements.txt```
  2. ```python3 flask_server.py```
