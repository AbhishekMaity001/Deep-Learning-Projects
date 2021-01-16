import tensorflow as tf
import numpy as np
import librosa

MODEL_PATH = 'model.h5'
NUM_SAMPLES_TO_CONSIDER = 22050 #SAMPLES 1 SECOND

class Keyword_Spotting_Service :

    model = None
    _mappings = [
        "bed",
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "four",
        "go",
        "happy",
        "house",
        "left",
        "marvin",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "sheila",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "wow",
        "yes",
        "zero"
    ]
    _instance = None # declairng a private variable

    def predict(self, file_path):
        # extract the MFCCs
        MFCCs = self.preprocess(file_path)  #segments , #coefficients

        # convert MFCCs array from 2d into 4d arrays ...bcoz of CNN -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make predictions
        predictions = self.model.predict(MFCCs) # 2d array ... [[smple1], [smple2], [smple3]...]
        predicted_index = int(np.argmax(predictions))
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft = 2048, hop_length=512):

        #load audio file
        signal, sr = librosa.load(file_path)

        # consticency in audio file...ie right amt of signals
        if len(signal) > NUM_SAMPLES_TO_CONSIDER :
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)

        return MFCCs.T




def Keyword_Spotting_Service_function():

    # implementing singleton class in python
    # ensure that we only have 1 instance of the keyword spotting service class
    if Keyword_Spotting_Service._instance is None :
        Keyword_Spotting_Service._instance = Keyword_Spotting_Service() # assign the instance of class to the class variable
        Keyword_Spotting_Service.model = tf.keras.models.load_model(MODEL_PATH) # loading the model
    return Keyword_Spotting_Service._instance

if __name__ == '__main__' :
    kss = Keyword_Spotting_Service_function()
    keyword1 = kss.predict('test/happy.wav')
    keyword2 = kss.predict('test/six.wav')
    keyword3 = kss.predict('test/down.wav')
    keyword4 = kss.predict('test/zero.wav')

    print(f"Predicted words are : {keyword1}, {keyword2}, {keyword3}, {keyword4}")



