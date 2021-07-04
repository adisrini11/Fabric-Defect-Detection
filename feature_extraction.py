from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
#from matplotlib import pyplot
from numpy import expand_dims
import cv2
import numpy as np

model=VGG16()

ixs=[2,5,9,13,17]
outputs=[model.layers[i].output for i in ixs]
model= Model(inputs=model.inputs,outputs=outputs) 
   
img=load_img(r'C:\Users\Aditya Srinivas\Final Year\try9.jpeg',target_size=(224,224))
    
img=img_to_array(img)
img=expand_dims(img,axis=0)
img=preprocess_input(img)
feature_maps=model.predict(img)

required = []

for fmap in feature_maps:
    fmap=np.squeeze(fmap,axis=None)
    fmap = fmap[:,:,1]
    fmap = cv2.resize(fmap,(224,224))
    required.append(fmap)

cv2.imshow("abc",required[0])
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("f5.png",required[0])
