import os
import librosa
import json

DATASET_PATH = 'audio'
JSON_PATH = 'extracted_data.json'
SAMPLES_TO_CONSIDER = 22050

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_ftt=2048) :

    # data dictionary
    data = {
        "mappings" : [], # mappings = ["on", "off", ...]
        "labels" : [], # target labels[0, 0, 1, 1, 1,...]
        "MFCCs" : [], # our inputs
        "files" : [] # "dataset/bed.wav"
    }

    # loop through all the subdir
    for i , (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)) :

        # ensure that we are not at the root level
        if dirpath is not dataset_path :

            # update mappings
            category = dirpath.split('\\')[-1]
            data["mappings"].append(category)
            print(f"Processing ===>  {category}")

            # looping through all the filenames
            for f in filenames :
                filepath = os.path.join(dirpath, f) # get the file path
                signal, sr = librosa.load(filepath)  # load the audio file
                if len(signal) >= SAMPLES_TO_CONSIDER : # check the audio file is atleast 1 sec

                    # making signals only 1 sec long if they are longer that 1 second
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    #extract the mfccs
                    MFCCs = librosa.feature.mfcc(signal,
                                                 n_mfcc=n_mfcc,
                                                 hop_length = hop_length,
                                                 n_fft = n_ftt)

                    # store data
                    data['labels'].append(i-1)
                    data['MFCCs'].append(MFCCs.T.tolist()) # librosa returns an nd array so we cast it to a list to store in json file
                    data['files'].append(filepath)
                    print(f"{filepath} : {i-1}")

    # store in json file
    with open(json_path, "w") as fp :
        json.dump(data, fp, indent=4)

if __name__ == "__main__" :
    prepare_dataset(DATASET_PATH, JSON_PATH)


