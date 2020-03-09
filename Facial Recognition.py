import os
import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read("trainer.yml")


label = {}
with open("labels.pickle", 'rb') as f:
    original_label = pickle.load(f)
    label = {v:k for k,v in original_label.items()} 

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print("Could not open camera")
    exit()
    
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
    for x,y,w,h in face:
##        print(x,y,w,h)
        ## Writing of facial images in grey scale (y-cord_start, y-cord_end)
        region_grey = gray[y:y+h, x:x+w] 
        ## Writing of facial image in color
        region_color = frame[y:y+h, x:x+w]

        #Deep Learning model Here -> Recognising
        id_, conf = recogniser.predict(region_grey)
        if conf >= 4 and conf <= 85:
##            print(id_)
##            print(label[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = label[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            
        Base_dir = os.path.dirname(os.path.abspath(__file__))
        temp = os.path.join(Base_dir, 'temp')
        img_item = os.path.join(temp, 'my-image' + str(count) + '.png')
        cv2.imwrite(img_item, region_color)
        count = count + 1

##      creating Blue rectangle around face
        color = (255, 0, 0) #Blue Color
        stroke = 2
        end_coordinate_x = x + y
        end_coordinate_y = y + h
        cv2.rectangle(frame, (x, y), (end_coordinate_x, end_coordinate_y),color, stroke)
        subitems = smile_cascade.detectMultiScale(region_grey)
        for ex,ey,ew,eh in subitems:
            cv2.rectangle(region_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    cv2.imshow('frame', frame)
    if (cv2.waitKey(20) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
