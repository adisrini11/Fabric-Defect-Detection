import cv2
import numpy as np
inimg = cv2.imread('s5.png',0)

def image_threshold(img):
    mu=np.mean(img)
    stdev=np.std(img)
    constant = 6.5
    t_min = mu-constant*stdev
    t_max = mu+constant*stdev
    img_threshold = np.zeros((img.shape[0],img.shape[1],1))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i,j]
            if t_min < pixel < t_max:
                img_threshold[i,j] = 0
            else:
                img_threshold[i,j] = 255
        
    print(f"mean: {mu}")
    print(f"stdev: {stdev}")
    return img_threshold

'''
op = image_threshold(inimg)
cv2.imshow("threshold image",op)
#cv2.imwrite("threshold_image.png",img_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("op5.png",op)
'''
