################### server libs ########################

from flask import Flask, request, send_file
from flask_restful import Resource, Api 
from werkzeug.utils import secure_filename

################### model libs ########################
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import cv2
import mrcnn.model as modellib
from mrcnnn import utils, cus_viz
from mrcnnn.config import Config
import tensorflow as tf


########## Init class labels for MASK RCNN #############
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


################### Init Path variables for MASK RCNN ###

# Local path to trained weights file
ROOT_DIR = os.getcwd()
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join("images")


############ Initialize model configuration #############

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Initialize a shared session for keras / tensorflow operations.
session = sess = tf.compat.v1.Session()
init = tf.global_variables_initializer()
session.run(init)

############ Load MASK RCNN ###########################
# Create model object in inference mode.



model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()


def predict(path):
    global model
    
    image = skimage.io.imread(path)
    results = model.detect([image], verbose=1)

    return (image, results[0])






############ Flask Server Initialization ###############
app = Flask(__name__)
api = Api(app)


class File(Resource):
    """
    Resource endpoint to handle file sharing. Will be used to get image from client and show Inference.
    """
    def post(self, name):
        # extracts the file from a POST request and saves it.
        f = request.files['file']
        #filename = secure_filename(f.filename)
        filename = "input.jpg"
        save_path = os.path.join(os.getcwd(), filename)
        f.save(save_path)
     
        image, r = predict(save_path)
        
        print(type(image), image.shape)
        
        cus_viz.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        #return "file processed and saved.", 201
        return send_file('output.jpg', attachment_filename='output.jpg', as_attachment=True)


api.add_resource(File, '/file/<string:name>')


if __name__ == '__main__':
    app.run('0.0.0.0', port = 5000, debug =True)