## Code in this file is involved in creating the images that will be fed to the deep neural network for training purposes
## This file contains preprocessing functions for DISFA, Kohn-Canade and MMI datasets.

import os
import numpy as np
import csv,cv2
from cv2 import CascadeClassifier, imread, imshow

target_aus = {6,12,1,4,15,2,5,26,7,20,23,9,15,16,12,14} #aus involved in emotions

#padding for face cropping
padding = 20

# writeCsvData(True)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Samples of how datasets would be processed
writeCsvData(True)
preprocessDISFA(save_images=False,target_aus=target_aus)
print('DISFA preprocessed')

writeCsvData(True)
preprocessKohn(save_images=True,target_aus=target_aus)
print('Kohn preprocessed')

writeCsvData(True)
preprocessMMI(save_images=False,target_aus=None)
print('MMI preprocessed')


########################################################
######      PREPROCESSING FOR MMI DATASET        #######
########################################################

#delete images from directory
def cleanDir(folder):
    import os, shutil
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

#returns a frame to AU dictionary from a session oao xml file
from xml.dom import minidom
def framesToAuDic(path_to_aoc_xml,target_aus = None):
    # parse an xml file by name
    aoc_xml = minidom.parse(path_to_aoc_xml)

    #calculate frames to au dictionary
    actionUnits = aoc_xml.getElementsByTagName('ActionUnit')
    clipData = {} #dictionary. The keys are the frames and the values are the au present in it

    for au in actionUnits:
        au_value  = au.attributes['Number'].value
        try: 
            if target_aus is not None:
                if int(au_value) not in target_aus:
                    continue
        except ValueError: #only allowing integer aus
            continue
       
        markers = au.getElementsByTagName('Marker')
        
        #find the range of usable keyframes --------------------------------------
        type1frame=0
        type2frame=0
        type3frame=0
        last_type0frame=0
        for m in markers:
            if m.attributes['Type'].value == "0" and m.attributes['Frame'].value != "1":
                last_type0frame=m.attributes['Frame'].value
            if m.attributes['Type'].value == "1":
                type1frame=m.attributes['Frame'].value
            if m.attributes['Type'].value == "2":
                type2frame=m.attributes['Frame'].value
            if m.attributes['Type'].value == "3":
                type3frame=m.attributes['Frame'].value
        type1frame=int(type1frame)
        type2frame=int(type2frame)
        type3frame=int(type3frame)
        last_type0frame=int(last_type0frame)       
        #if either of these was not found then don't use these keyframes
        if(type1frame*type2frame*type3frame*last_type0frame == 0):
            continue
        start_valid_frame = type1frame + (type2frame-type1frame)//2
        end_valid_frame = type3frame + (last_type0frame-type3frame)//2
        #-----------------------------------------------------------------------------
        
        #choose specific frames that have slightly different expressions but all corresponding to the activated AUs
        for frame in range(start_valid_frame+2,type2frame+2,2): #use every other frame in the valid sequence
            if frame in clipData:
                    clipData[frame].append(au_value)
            else:
                clipData[frame] = [au_value]
        for frame in range(type3frame-1,end_valid_frame-2,2): #use every other frame in the valid sequence
            if frame in clipData:
                    clipData[frame].append(au_value)
            else:
                clipData[frame] = [au_value]
    return clipData
#saves the cropped faces of the set of requested keyframes of the path_to_video in the save_dir


def saveCroppedImages(path_to_video, save_dir, keyframes, resizeDim = None):

        
    keyframes = list(map(int,keyframes))
    
#     classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
    
    video = cv2.VideoCapture(path_to_video)
   
    success,image = video.read()
    
    if video is None or success == False:
        raise ValueError("unable to read " + path_to_video)
        
    count = 1
    while success:
        if count in keyframes: 
            
            #CROP THE TOP TO HACK BUG THAT DOESNT DETECT SOME FACES IN MMI!!!!
            image = image[:-50,:]
            
            #detect main face bounding box
            bboxes = classifier.detectMultiScale(image)
            
            #ADD PADDING TO CV2 FACE DETECTOR
            if len(bboxes)<1:
                #try rotating the image
                image = np.rot90(image,axes=(1, 0))
                bboxes = classifier.detectMultiScale(image)
                
            if len(bboxes)<1: #if after roation the face detector is not able to find a bounding box then ignore the image
                print("Couldn't find a face on " + save_dir + "/" + path_to_video[-12:-4]+ "_frame%d.png" % count )
            else: 
                x, y, w, h = bboxes[0]

                #LEAVE FRAME AROUND IMAGE
                crop_img = image[y-padding:y+h+padding, x-padding:x+w+padding]

                if (crop_img.shape[0]*crop_img.shape[1]*crop_img.shape[2])==0:
                    print(save_dir + "/" + path_to_video[-12:-4]+ "_frame%d.png" % count + "'s cropped version was empty")
                else:
                    if resizeDim is not None:
                        width, height = resizeDim
                        crop_img = cv2.resize(crop_img,(width,height))

                    #black and white image
#                     crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#                     crop_img = np.stack((crop_img,)*3, axis=-1)
                    cv2.imwrite(save_dir + "/" + path_to_video[-12:-4]+ "_frame%d.png" % count, crop_img)     # save frame as PNG file      
        success,image = video.read()
        count += 1

#save image pointers in csv file with au labels

#POPULATE CSV_DATA_ROWS
def writeCsvData(includeTitle,base_file_name=None,clipData=None,dataset=None):
    if includeTitle == True:
        csvFile = open('facs.csv', 'w+')
        writer = csv.writer(csvFile)
        writer.writerows([['filePath', 'actionUnits']])
        csvFile.close()
        return
    
    #title=False
    csvFile = open('facs.csv', 'a+')
    writer = csv.writer(csvFile)
    csv_data_rows = []
    
#     base_file_name = 'facs_processed_images/' + dataset + '/'+ base_file_name

    
    for frame in clipData:
        csv_data_rows.append([base_file_name + "_frame%d.png" % frame,','.join(map(str, clipData[frame]))])
    writer.writerows(csv_data_rows)
    csvFile.close()

#PROCESS MMI IMGAES. SAVE CROPPED IMAGES AND UPDATE CSV FILE. RUN THIS FIRST AS IT ALSO CREATES THE CSV FRAME TITLE
#TODO: not only save the peak image but also its sourroundings (use the oao coding to know upto which frame to save)
def preprocessMMI(save_images=True,target_aus=None):
    save_dir = 'facs_processed_images/mmi'
    session_dir = 'mmi-oao-page10/Sessions/501'
    if save_images:
        cleanDir(save_dir)

    # target_au = {1,2,4,5,6,7,9,12,14,15,16,20,23,26}
    mmi_dir = 'mmi-oao/Sessions'

    session_folders = next(os.walk(mmi_dir))[1]
    
    for session in session_folders:
        session_dir = mmi_dir + "/" + session
        #find oao xml file
        files = os.listdir(session_dir)
        oao_file = None
        avi_file = None

        #save cropped images in save_dir
        for f in files:
            if f[-12:] =="oao_aucs.xml":
                oao_file=f
            if f[-3:] == "avi":
                avi_file=f
        clipData = framesToAuDic(session_dir + '/' + oao_file,target_aus=target_aus)
        if save_images:
            saveCroppedImages(session_dir + '/' + avi_file, save_dir, set(clipData.keys()), resizeDim = (224,224))
        writeCsvData(False,base_file_name=avi_file[:-4],clipData=clipData,dataset='mmi')


#############################################################
######      PREPROCESSING FOR KOHN-CANADE DATASET     #######
#############################################################
def preprocessKohn(save_images=True,target_aus=None):
    csvFile = open('facs.csv', 'a+')
    writer = csv.writer(csvFile)
    if save_images:
        cleanDir('kohn_emotions/')
    csv_data_rows = []
    neutral_counter = 0

    sessions = next(os.walk('cohn/cohn-kanade-images'))[1]

    for session in sessions:
        session_dir = 'cohn/cohn-kanade-images/'+session
        session_folders = next(os.walk(session_dir))[1]
        for folder in session_folders:
            files = os.listdir(session_dir + "/" + folder)
            files = list(filter(lambda x: x !=".DS_Store",files))
            #only the last third of the images will be used as the rest are neutral expresssions (transition)
            
            files = sorted(files)
            index=0
            for file in files:
                neutral_counter +=1
                index+=1
                if not((index > (len(files) * .4) and index % 3 == 0)or (index ==1  and neutral_counter % 8 == 0)):
                    continue

                #detect main face bounding box and save cropped image
                if save_images:
                    image = cv2.imread(session_dir + "/" + folder + "/" + file)
                    print(session_dir + "/" + folder + "/" + file)
                    bboxes = classifier.detectMultiScale(image)
                    x, y, w, h = bboxes[0]

                    #LEAVE FRAME AROUND IMAGE
                    crop_img = image[y-padding:y+h+padding, x-padding:x+w+padding]

                    #square image
                    if (crop_img.shape[0]*crop_img.shape[1]*crop_img.shape[2])==0:
                        continue
                    crop_img = cv2.resize(crop_img,(224,224))

                    #black and white image
#                     crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#                     crop_img = np.stack((crop_img,)*3, axis=-1)
                    cv2.imwrite("kohn_emotions/" + file, crop_img,[cv2.IMWRITE_PNG_COMPRESSION, 0]) #No compression, large size but fast decompression

#                 #CREATE CSV INFORMATION
#                 #read aus and 
#                 facs_dir = 'cohn/FACS/' + file[:4] + '/' + file[5:8]
#                 facs_file = facs_dir + '/' + os.listdir(facs_dir)[0]
#                 with open(facs_file) as facs:
#                     text = facs.read()
#                     text = list(filter(lambda x: x!='',text.split(' ')))
#                     text= text[0::2]
#                     aus = list(map(lambda x: int(float(x)),text))
# #                     aus = list(filter(lambda x: x in target_aus,aus))
#                 if len(aus)>0:
# #                     csv_data_rows.append(["facs_processed_images/cohn/" +file,','.join(map(str, aus))])
#                     csv_data_rows.append([file,','.join(map(str, aus))])

    writer.writerows(csv_data_rows)
    csvFile.close()

########################################################
######      PREPROCESSING FOR DISFA DATASET     #######
########################################################
def reduceClipDataDISFA(clip_data):
    keys = list(clip_data.keys())
    keys.sort()
    last_aus = None
    reduced_clip_data = {}
    for k in keys:
        if clip_data[k]==last_aus: #take 1 out of every 20th frame
            if k%20==0:
                reduced_clip_data[k]=clip_data[k]
        else:
            reduced_clip_data[k]=clip_data[k]
        last_aus = clip_data[k]
            
    return reduced_clip_data

#save images and csv information for the DISFA dataset
def preprocessDISFA(save_images=True,target_aus=None):
    if save_images:
        cleanDir('facs_processed_images/disfa')
    disfa_dir = 'disfa'

    label_folders = next(os.walk(disfa_dir + '/labels'))[1]
    for label_folder in label_folders:
        facs_files = os.listdir(disfa_dir + '/labels/' + label_folder)
        clip_data = {}
        for file in facs_files:
            with open(disfa_dir + '/labels/' + label_folder + '/' + file, 'r') as f:
                au = int(file[-6:-4].replace('u',''))
                if target_aus is not None:
                    if au not in target_aus:
                        continue
                for line in f:
                    line = line.replace('\n','')
                    line = line.split(',')
                    line = [int(line[0]),int(line[1])]
                    frame = int(line[0])
                    intensity = int(line[1])

                    if(intensity<1):
                        continue

                    #add to frame->au dic
                    if frame in clip_data:
                            clip_data[frame].append(au)
                    else:
                        clip_data[frame] = [au]
        clip_data = reduceClipDataDISFA(clip_data)

        left_clip_data = {}
        right_clip_data = {}
        #alternate taking frames from left and right camera
        counter = 0
        for k in clip_data.keys():
            if counter%2==0:
                left_clip_data[k]=clip_data[k]
            else:
                right_clip_data[k]=clip_data[k] 
            counter+=1
            
#         save images in file
        if save_images:
            saveCroppedImages('disfa/videos_right/RightVideo' + label_folder + '_comp.avi', 'facs_processed_images/disfa/', right_clip_data, resizeDim = (224,224))
            saveCroppedImages('disfa/videos_left/LeftVideo' + label_folder + '_comp.avi', 'facs_processed_images/disfa/', left_clip_data, resizeDim = (224,224))

#         save csv info
        writeCsvData(False,base_file_name=label_folder[2:]+"_comp",clipData=right_clip_data,dataset='disfa')
        writeCsvData(False,base_file_name=label_folder[2:]+"_comp",clipData=left_clip_data,dataset='disfa')


