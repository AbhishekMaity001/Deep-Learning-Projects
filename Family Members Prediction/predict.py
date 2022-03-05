
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



class familyMembers :
    def __init__(self,filename):
        self.filename = filename

    def predictfamilyMembers(self):
        model = load_model(r'D:\Data Science\deeplearning-model files\family predictn model files\Epoch100.h5')
        imgname = self.filename
        test_image = image.load_img(imgname,target_size=(224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        result = model.predict(test_image)

        if (np.argmax(result) == 0):
            #print("This is Abhishek!! ")
            return [{"image": "This is Abhishek!!"}]

        elif (np.argmax(result) == 1):
            #print('This is Ashirbad!! ')
            return [{"image": "This is Ashirbad!!"}]

        elif (np.argmax(result) == 2):
            #print('This is Maa !!')
            return [{"image": "This is Maa!!"}]

        else:
            #print('This is Ria !!')
            return [{"image": "This is Ria!!"}]
