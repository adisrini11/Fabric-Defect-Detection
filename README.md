# Fabric-Defect-Detection
Previous fabric defect detection techniques were not robust and for each kind of fabric, a different technique had to be employed. This project employs techniques which are robust and works for both patterned and unpatterned fabric.

The system uses a VGG16 pre trained CNN model which is 16 layers deep and is pre trained with more than a million images from the ImageNet database. This pre trained CNN model is used to extract features from images of fabric pieces. The features from the shallow layers of the CNN are used for further processing as the features from such layers contains spatial information.

The system is designed on the idea that the defects in the fabric form a sparse part and we can consider the defect as an object and the non defective regions in the fabric is redundant and can be considered as the background.

This idea allows us to use a modified version of a statistical tool, Prinicipal Component Analysis (PCA), called the Robust Prinicipal Component Analysis (RPCA), to separate the object (defective region) from the background (non defective region).

To further enhance the output and make it more accurate, a thresholding function was designed. The function classified pixels in the output of RPCA as belonging to the defective or non-defective region.

![](/Inputs.png)

![](/Final%20Output)
