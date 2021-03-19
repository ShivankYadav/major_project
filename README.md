# Major_project
This repository contains contents of major project 2021

## Steps to run
  **Docker Commands**
  1. Build the image from dockerfile using ```docker image build -t cudachen/mask-rcnn-docker .```
  2. Build the container from the image using ```docker run -it -p 8888:8888 -p 6006:6006 --name mrcnn_flask -v $(pwd):/host cudachen/mask-rcnn-docker```
  3. If the container is already created, use the below command to get it up and running:
        1. ```docker container start mrcnn_flask```
        2. ```docker container exec -it mrcnn_flask /bin/bash```
  **Commands to run inside the container**
  (Everytime commands)
  1. ```cd /host```
  
  (Only to be run If created the container for the first time)
  1. ```pip3 install -r requirements.txt```
