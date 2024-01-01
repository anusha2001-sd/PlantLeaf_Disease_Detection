from tensorflow.python.keras.utils import np_utils
import keras
import numpy as np
import tensorflow as tf
from os import path, listdir
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow. keras.layers import Input,GlobalMaxPooling2D,Dense
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

label_map = {'DOWNY_MILDEW': 0,
 'HEALTHY': 1,
 'LEAFY_MINOR': 2}
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image and converts  into an object 
        that can be used as input to a trained model, returns an Numpy array.

        Arguments
        ---------
        image_path: string, path of the image.
    '''
    
    im = load_img(image_path
                   , target_size=(224,224))
    im =img_to_array(im)
    im = np.expand_dims(im, axis=0)
    
    return im
def get_key(val): 
    for key, value in label_map.items(): 
         if val == value: 
             return key 
def process(path):
    model = load_model("Final_model.h5")
    image = process_image(path)

    prediction = model.predict(image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    print("prediction",prediction)
    
    return get_key(np.argmax(prediction)),confidence
#print("Predicted=",process("Data/LEAFY_MINOR/IMG-20230302-WA0009_jpg.rf.c02b0b4545b6cfad33c188505c58579e.jpg"))
    
