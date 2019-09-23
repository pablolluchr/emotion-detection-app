## Code in this file is involved in creating the images that will be fed to the deep neural network for training purposes
## This file contains preprocessing functions for BP4D dataset
import os
import csv
from cv2 import CascadeClassifier, imread, resize, imwrite


train_csvFile = open('train_facs.csv', 'w+')
train_writer = csv.writer(train_csvFile)
train_writer.writerows([['filePath', 'actionUnits']])
    
val_csvFile = open('val_facs.csv', 'w+')
val_writer = csv.writer(val_csvFile)
val_writer.writerows([['filePath', 'actionUnits']])

validation_split = {"F005","F017","M008","M012"} #sessions that will be put into validation 

train_csv_data_rows = []
val_csv_data_rows = []
#use the au_occ csv files to iterate through the sessions
files = os.listdir('AUCoding/AU_OCC')
sessions_dir = "dataset/www.cs.binghamton.edu/~gaic/BP4D/Sequences(2D+3D)"

def addToCSV(name,aus):
    if name[:4] in validation_split:
        val_csv_data_rows.append([image_path,aus])
    else:
        train_csv_data_rows.append([image_path,aus])

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

#return comma separated list of action units from index encoded vector
def getAus(raw):
    aus = ""
    for i in range(len(raw)):
        if raw[i] == '1':
            aus += "{},".format(i+1) 
    return aus[:-1]

#cleanDir('facs_processed_images/bp4d/')


for f in files:
    # csv_reader = csv.reader('bp4d/AUCoding/AU_OCC/{}'.format(f), delimiter=',')
    if f.endswith(".DS_Store"):
        continue
    with open('bp4d/AUCoding/AU_OCC/{}'.format(f)) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip header
        for row in reader:
            processed_image_name = "{}_{}_{}.png".format(f[:4],f[5:7],row[0].zfill(4))
            task_image_dir = "{}/{}/{}/{}.jpg".format(sessions_dir,f[:4],f[5:7],row[0].zfill(3))
            aus = getAus(row[1:])

            #add to csv and continue if image is already saved
            if os.path.exists("facs_processed_images/{}.png".format(processed_image_name)):
                addToCSV(processed_image_name,aus)
                print("{}.png is already cropped".format(processed_image_name))
                continue
            
            #READ IMAGE AND CROP FACE FROM IT
            img = imread(task_image_dir)
            if img is None: #try using 0000 numbers instead of 000
                task_image_dir = "bp4d/sessions/{}/{}/{}.jpg".format(f[:4],f[5:7],row[0].zfill(4))
                img = imread(task_image_dir)
                
            #crop face
            classifier = CascadeClassifier('haarcascade_frontalface_alt2.xml')
            bboxes = classifier.detectMultiScale(img)
            padding = 30
            x, y, w, h = bboxes[0]
            img = img[y-padding:y+h+padding, x-padding:x+w+padding]
            if len(img)==0 or len(img[0]) == 0:
                print("Couldn't crop {}.png".format(processed_image_name))
                continue
            img = resize(img,(224,224))
            
            #save image file and update csv
            image_path = "facs_processed_images/bp4d/{}_{}_{}.png".format(f[:4],f[5:7],row[0].zfill(4))
            imwrite("facs_processed_images/{}".format(processed_image_name),img)
            addToCSV(processed_image_name,aus)
            

    
train_writer.writerows(train_csv_data_rows)
val_writer.writerows(val_csv_data_rows)
train_csvFile.close()
val_csvFile.close()
