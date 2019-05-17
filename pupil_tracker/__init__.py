from keras.models import load_model

model = load_model('./models/model.h5')
classes = ['bottom_left', 'bottom_right', 'normal', 'top_left', 'top_right']

import pupil_tracker.predict
import pupil_tracker.preprocess
