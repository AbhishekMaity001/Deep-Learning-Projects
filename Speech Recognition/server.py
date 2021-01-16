""""
flask server running on port 5000 or any

client --> POST request --> server -> prediction back to client

"""
import random
import os
from flask import Flask, request, jsonify


from keyword_spotting_service import Keyword_Spotting_Service_function

app = Flask(__name__)


# route the incoming request to the api endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # get audio file and save this
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 10000))
    audio_file.save(file_name)

    # invoke keyword spotting service
    kss = Keyword_Spotting_Service_function()

    #prediction
    predicted_keyword = kss.predict(file_name)

    # remove the audio file temp stored
    os.remove(file_name)

    # send back the predicted keyword in json format
    data = {"keyword" : predicted_keyword}
    return jsonify(data)

if __name__ == "__main__" :
    app.run(debug=False)


