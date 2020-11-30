from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
import re
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect , url_for, request, render_template
from werkzeug.utils import  secure_filename

#Defining the Flask app
app = Flask(__name__)

# Model path saved with keras model.save() method
Model_path = 'model_inception_colab.h5'
# Now loading the model
model  = load_model(Model_path)

def model_predict(image_path,model):
    print(image_path)
    img = image.load_img(image_path, target_size=(224,224))

    # Converting the loaded image to an array
    x = image.img_to_array(img)
    # Scaling down 0-1
    x = x/255.0
    # Expanding the dimensions of the image along the x axis
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1) # Taking only the maximum value in the output
    if preds == 0 :
        preds = 'This leaf is having Bacterial Spot'

    elif preds == 1 :
        preds = 'This leaf is having Early Blight'

    elif preds == 2:
        preds = 'This leaf is Health Leaf :) '

    elif preds == 3 :
        preds = 'This leaf is having Late Blight '

    elif preds == 4 :
        preds = 'This leaf is having Leaf Mold '

    elif preds == 5 :
        preds = 'This leaf is having Septoria Leaf Spot '

    elif preds == 6 :
        preds = 'This leaf is having Two Spotted Spider Mite '

    elif preds == 7 :
        preds = 'This leaf is having Target Spot '

    elif preds == 8 :
        preds = 'This leaf is having Mosaic Virus '

    else :
        preds = 'This leaf is having Yellow Curl Virus '


    return preds

@app.route('/',methods=['GET'])
def index() :
    # This is main home page
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

if __name__ == '__main__':
    app.run(port=5001,debug=True)