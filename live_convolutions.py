#TRY ON ONE EXAMPLE

import numpy as np


from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from keras import backend as K
from cv2 import CascadeClassifier, imread, resize


#LOAD CNN

#functions needed to load cnn
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
dependencies = {'recall': recall,'precision': precision,}
model = load_model("action_unit_cnn.hdf5", custom_objects=dependencies)



from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
# model.summary()
 
layer_outputs = [layer.output for layer in model.layers[1:18]] 
# layer_outputs_maxpooling = [MaxPooling2D()(layer.output) for layer in model.layers[1:18]] 


# # Extracts the outputs of the top 12 layers
activation_model = Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
# activation_model_maxpooling = Model(inputs=model.input, outputs=layer_outputs_maxpooling) # Creates a model that will return these outputs, given the model input

image_width = 50
def makeDisplayGrid(img):
    
    start_time = time.time()

    activations = activation_model.predict(img)

    #display a row of images per layer 
    rows = len(activations)
    cols = 16 #image per row
    display_grid = np.zeros((rows * image_width, cols * image_width))


    for index, layer_activation in zip(range(rows), activations): # Displays the feature maps
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1] 
        
        sample_images = layer_activation[:,:,:,:cols]

        for col in range(cols):
            try:
                base_image = sample_images[0,:,:,col]
                base_image -= base_image.mean()
                base_image /= base_image.std()
                base_image *=64
                base_image +=128
                base_image = np.clip(base_image, 0, 255).astype('uint8')
                base_image = cv2.resize(base_image,(image_width,image_width))
                display_grid[index*image_width:(index+1)*image_width,col*image_width:(col+1)*image_width]=base_image
            except:
                continue
    # print("Inference + display_grid calculation time: " + str(time.time()-start_time))
    return display_grid

cv2.namedWindow('DeepEmotion', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    try:
        crop_reduction = 3
        frame_face = cv2.resize(frame,(frame.shape[1]//crop_reduction,frame.shape[0]//crop_reduction))
        bboxes = classifier.detectMultiScale(frame_face)
        padding = 30
        x, y, w, h = bboxes[0]
        (x, y, w, h) = (x*crop_reduction, y*crop_reduction, w*crop_reduction, h*crop_reduction) 
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)

        #crop face for inference and put it in inference queue
        img = frame[y-padding:y+h+padding, x-padding:x+w+padding]
    except:
        img = frame
    img = cv2.resize(img,(224,224))
    img = np.expand_dims(img, axis=0)
    
    display = np.array(makeDisplayGrid(img)) / 255.
    # display = cv2.applyColorMap(display, cv2.COLORMAP_VIRIDIS)
    # display = cv2.resize((display),(20,20))
    # print(display)
    cv2.resizeWindow('DeepEmotion', 1000,1000)
    cv2.imshow('DeepEmotion',display)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break


