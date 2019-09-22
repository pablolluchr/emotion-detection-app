import logging
import time
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from multiprocessing import Process, Queue

from tensorflow.keras.models import load_model
from keras import backend as K


def inferEmotion(frame_queue,emotion_queue):

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

    dependencies = {
        'recall': recall,
        'precision': precision,
    }
    aus_at_index = [1,10,12,14,15,17,2,20,23,24,26,27,4,5,6,7,9]
    aus_dic = {0:"Neutral face",1:"Inner brow raiser",2:"Outer brow raiser",4:"Brow lowerer",5:"Upper lid raiser",6:"Cheek raiser",7:"Lid tightener",8:"Lips toward each other",9:"Nose wrinkler",10:"Upper lip raiser",11:"Nasolabial deepener",12:"Lip corner puller",13:"Sharp lip puller",14:"Dimpler,buccinator",15:"Lip corner depressor",16:"Lower lip depressor",17:"Chin raiser",18:"Lip pucker",19:"Tongue show",20:"Lip stretcher",21:"Neck tightener",22:"Lip funneler",23:"Lip tightener",24:"Lip pressor",25:"Lips part",26:"Jaw drop",27:"Mouth stretch",28:"Lip suck"}

    model = load_model("action_unit_cnn.hdf5", custom_objects=dependencies)
    model.predict(np.zeros((1,224,224,3)))
    emotions = np.load("emotion_predictions_neutral_only_peaks.npy",allow_pickle=True)
    # emotions = list(map(lambda x: [np.array(x[0])*1.4,x[1]],emotions))
    emotions = list(map(lambda x: [np.array(x[0])+0.1,x[1]],emotions))
    past_emotions = []

    #load image

    while True:
        img = frame_queue.get(block=True, timeout=None)
        
        start = time.time()
        preds = model.predict(img)[0]

        thresholds = np.ones(len(preds)) * 0.4
        binary_preds = list(map(lambda x: 1 if x[0]>x[1]else 0,zip(preds,thresholds)))
        present_aus = list(filter(lambda x: x>0,np.multiply(binary_preds,aus_at_index)))
        present_aus_description = list(map(lambda x: aus_dic[x],present_aus))

        #find the most popular emotions using cosine similarity to _emotions_
        dist = list(map(lambda x: [cosine(preds,x[0]),x[1]],emotions))

        #find the most popular emotions using euclidean distance
        # dist = list(map(lambda x: [np.linalg.norm(preds-x[0]),x[1]],emotions))
        dist.sort(key = lambda x: x[0])
        dist = dist[:1]
        popular_emotions = list(map(lambda x: x[1],dist))

        emotions_dic = {0:"neutral", 1:"anger", 2:"contempt", 3:"disgust", 4:"fear", 5:"happiness", 6:"sadness", 7:"surprise"}
        top_emotion = max(set(popular_emotions), key=popular_emotions.count)

        #if no aus activated then set emotion back to neutral
        if len(present_aus) == 0:
            top_emotion = 0

        # reduce noise by making the top emotion the most popular top emotion over the past 5 inferences
        if len(past_emotions)>=4:
            past_emotions.pop()
        past_emotions.insert(0,top_emotion)
        top_emotion = max(set(past_emotions), key=past_emotions.count)

        #if there's not a clear predominant emotion then set to neutral
        if past_emotions.count(top_emotion) < 3:
            top_emotion= 0

        inference_time = ("Inference time: " + str(time.time() - start))

        try:
            emotion_queue.put_nowait((emotions_dic[top_emotion],present_aus_description,inference_time))
        except:
            pass



def overlay_transparent(background, overlay, x, y):

    """combine alpha image at position x,y of background"""

    background_width = background.shape[1]
    background_height = background.shape[0]
    if x >= background_width or y >= background_height:
        return background
    h, w = overlay.shape[0], overlay.shape[1]
    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]
    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]
    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )
    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image
    return background
        
        

if __name__ == "__main__":
    frame_queue = Queue(maxsize=1)
    emotion_queue = Queue(maxsize=1)
    p = Process(target=inferEmotion, args=(frame_queue,emotion_queue))
    p.start()
    
    cap = cv2.VideoCapture(0)
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion = "neutral"
    aus_array =[]
    inference_time = ""
    
    cv2.namedWindow('DeepEmotion', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('DeepEmotion', 600,480)
    while(True):
        ret, frame = cap.read()

        try:
            #detect face and draw bounding box
            crop_reduction = 3
            frame_face = cv2.resize(frame,(frame.shape[1]//crop_reduction,frame.shape[0]//crop_reduction))
            bboxes = classifier.detectMultiScale(frame_face)
            padding = 30
            x, y, w, h = bboxes[0]
            (x, y, w, h) = (x*crop_reduction, y*crop_reduction, w*crop_reduction, h*crop_reduction) 
            cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,0),2)

            #crop face for inference and put it in inference queue
            img = frame[y-padding:y+h+padding, x-padding:x+w+padding]
            img = cv2.resize(img,(224,224))
            img = np.expand_dims(img, axis=0)
            if frame_queue.empty():
                frame_queue.put_nowait(img)

            #check emotion queue for emotion and aus and write them on output image
            if not emotion_queue.empty():
                emotion,aus_array,inference_time = emotion_queue.get_nowait()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(frame,(x,y-50),(x+h,y),(0,255,0,0.3),-1)
            cv2.putText(frame,emotion,(x,y-10), font, 1.5,(0,0,0),2,cv2.LINE_AA)
            offset = 0
            for au in aus_array:
                cv2.putText(frame,au,(50,50 + offset), font, 1,(0,0,0),2,cv2.LINE_AA)
                offset+=30
            emoji = cv2.imread("emojis/{}.png".format(emotion),cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji,(70,70))
            frame = overlay_transparent(frame,emoji,x+w-60,y-60)
        except:
            pass

        #show frame on screen
        frame = cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
        cv2.imshow('DeepEmotion',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    p.terminate()
    cap.release()
    cv2.destroyAllWindows()