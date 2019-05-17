from . import model # , classes
import numpy as np
from keras.preprocessing import image
import cv2

def prediction(img):
    img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
    img = np.expand_dims(img, axis=0)
    img = np.resize(img, (1, 64, 64, 3))
    result = model.predict(img)
    # return result[0], classes[result[0].argmax(axis=0)]
    return result[0], result[0].argmax(axis=0)
