# Major_project
This repository contains contents of major project 2021

## Steps to run
  **Docker Commands**
  1. Build the image from dockerfile using ```docker pull cudachen/mask-rcnn-docker```
  2. If you have already created the container before, you can follow step 4 directly.
  3. Build the container from the image using ```docker container run -it --network host --name mrcnn_flask -v $(pwd):/host cudachen/mask-rcnn-docker```. A new root container in bash mode shall open. You can enter ```exit``` to close the container after using it. Skip step 4.
  4. If the container is already created or to start the pre created container, use the below command to get it up and running:
        1. ```docker container start mrcnn_flask```
        2. ```docker container exec -it mrcnn_flask /bin/bash```
        
  **Commands to run inside the container**
  (Everytime commands)
  1. ```cd /host```
  
  (Only to be run If created the container for the first time)
  1. ```pip3 install -r requirements.txt```
  2. ```python3 flask_server.py```
  
  **To run the project**
  Simply open UI.html and you would be good to go. To close the server, open the terminal window and press Ctrl + C. After that you can enter ```exit``` to leave the root instance.
