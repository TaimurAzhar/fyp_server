from flask import Flask
from flask import jsonify
from flask import request
import json as json
from flask_cors import CORS, cross_origin


import base64
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'CONTENT-TYPE'


def get_model():
    global model
    model = load_model('gen.h5')
    print('* Model Loaded!')


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    # image = tf.image.resize(image, [target_size[0], target_size[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # image = image.resize(target_size)
    image = np.array(image)
    print("Print the shape")
    print(image.shape)
    # image = np.expand_dims(image, axis=0)
    # print("Print the shape after expansion")
    # print(image.shape)
    return image


print("* Loading Keras model....")
get_model()


@tf.function()
def resize(x_i, y_i, y_j, height, width):
    x_i = tf.image.resize(x_i, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    y_i = tf.image.resize(y_i, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    y_j = tf.image.resize(y_j, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return x_i, y_i, y_j


def normalize(x_i, y_i, y_j):
    
    x_i = (x_i / 127.5) - 1
    y_i = (y_i / 127.5) - 1
    y_j = (y_j / 127.5) - 1

    return x_i, y_i, y_j

@app.route("/predict", methods=["POST"])
def predict():
    images = request.get_json(force=True)
    x_i = images['x_i']
    y_i = images['y_i']
    y_j = images['y_j']

    print("\n ByteString \n")
    # print(encoded)

    x_i = base64.b64decode(x_i)
    y_i = base64.b64decode(y_i)
    y_j = base64.b64decode(y_j)

    x_i = img_to_array(Image.open(io.BytesIO(x_i)))
    y_i = img_to_array(Image.open(io.BytesIO(y_i)))
    y_j = img_to_array(Image.open(io.BytesIO(y_j)))


    # print(x_i.shape)
    x_i = tf.cast(x_i, tf.float32)
    y_i = tf.cast(y_i, tf.float32)
    y_j = tf.cast(y_j, tf.float32)
    # x_i = preprocess_image(x_i, target_size=(256,256))
    # y_i = preprocess_image(y_i, target_size=(256,256))
    # y_j = preprocess_image(y_j, target_size=(256,256))

    x_i, y_i, y_j = resize(x_i, y_i, y_j, 256, 256)
    x_i, y_i, y_j = normalize(x_i, y_i, y_j)
    print(y_i.shape)

    prediction = model([x_i, y_i, y_j])

    # cv2.imwrite('static/new_human.jpeg', y_j)   
    # response = {
    #     'prediction' : {
    #         'dog' : prediction[0][0],
    #         'cat' : prediction[0][1]
    #     }
    # }
    # return jsonify(response)
    return "works"

