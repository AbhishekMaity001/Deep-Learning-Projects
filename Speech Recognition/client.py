import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE = "test/down.wav"

if __name__ == "__main__" :
    audio_file = open(TEST_AUDIO_FILE, "rb") #getting the audio files as binary file
    values = {"file" : (TEST_AUDIO_FILE, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword is : {data['keyword']}")