import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


BASEDIR = os.path.dirname(os.path.abspath(__file__))

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"images")
categories = ["Zuriel","Velicia"]
training_data = []

for category in categories:
    path = os.path.join(DATADIR, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            IMG_SIZE = 80
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass
        
random.shuffle(training_data)
for sample in training_data:
    print(sample[1])

X = []
Y = []
for features, label in training_data:
    X.append(features)
    Y.append(label)

x = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

##pickle_in = open("X.pickle","rb")
##x = pickle.load(pickle_in)
##print(x[1])

