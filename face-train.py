import os
import cv2
from PIL import Image
import numpy as np
import pickle

## get the file address of this python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
image_dir = os.path.join(BASE_DIR, "images")
print (image_dir)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

#Train facial recogniser
recogniser = cv2.face.LBPHFaceRecognizer_create()


x_train = []
y_labels = []
current_id = 0
label_ids = {}

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"): #check directory for images
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id = current_id + 1
            id_ = label_ids[label]
            print(label_ids)
            #y_labels.append(label) # some number
            #x_train.append(path) # verify image, turn into NUMPY array and make it gray

            pil_image = Image.open(path).convert("L") # make it grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8") # transform image into numpy array
##            print(type(image_array))
            face = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=2)

            for x,y,w,h in face:
                region_of_interest = image_array[y:y+h, x:x+w]
                x_train.append(region_of_interest)
                y_labels.append(id_)


##print(y_labels)
##print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

x_train = x_train
y_labels = np.array(y_labels)
recogniser.train(x_train, y_labels)
recogniser.save("trainer.yml")

##43:52 / 1:06:23
##pip3 install opencv-contrib-python==3.4.6.27

























    
