import feature_extraction
import rpca
import thresholding
#import bounding_box
import cv2
from tensorflow.keras.preprocessing.image import load_img

image = load_img(r'C:\Users\Aditya Srinivas\Final Year\pn.jpg',target_size=(224,224))

feature_map = feature_extraction.extraction(image)

low_rank,sparse = rpca.run(feature_map)

output = thresholding.image_threshold(sparse)
#bounded = bounding_box.draw(output)
cv2.imshow("output",output)
cv2.waitKey()
cv2.destroyAllWindows()