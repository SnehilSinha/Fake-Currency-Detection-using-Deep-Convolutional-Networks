import os
import sys
import cv2

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

MODEL_PATH = 'models/counterfeit.h5'    # Model saved with Keras model.save()
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')

def model_predict(img, model):
    #img = img.convert('RGB')
    gray = cv2.imread(".\\uploads\\image.png", cv2.IMREAD_GRAYSCALE)
    high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    blur = cv2.medianBlur(gray, 17)
    
    edge_img = cv2.Canny(blur,lowThresh*1.5,high_thresh*1.5)
    pts = np.argwhere(edge_img>0)
    try:
        y1,x1 = pts.min(axis=0)
        y2,x2 = pts.max(axis=0)
    except ValueError:
        print (img)
        pass
            
    cropped = gray[y1:y2, x1:x2]
    blur_cropped = cv2.medianBlur(cropped, 7)
    blur_cropped = cv2.resize(blur_cropped , (224, 224))
    img=np.array(blur_cropped)
    x = np.array(img).reshape(-1,224,224,1)
    
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        
        #preds[0][0]=0.00031593
        print("pred= ",preds)
        print("pred[0][0]= ",preds[0][0])

        if (preds[0][0]) == 0.0:
            return jsonify(result="Fake!")
        else:
            return jsonify(result="Real")

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)

    # Serve the app with gevent
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()
