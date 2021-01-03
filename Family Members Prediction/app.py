from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from Helper.utils import decodeImage
from predict import familyMembers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp :
    def __init__(self):
        self.filename = "sampleImage.png"
        self.classifier = familyMembers(self.filename)


@app.route("/",methods = ["GET"])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
@cross_origin()
def predictRoute() :
    image = request.json['image']
    decodeImage(image,clp.filename)
    result =clp.classifier.predictfamilyMembers()
    return jsonify(result)

clp = ClientApp()

if __name__ == '__main__' :
    app.run(debug=True)