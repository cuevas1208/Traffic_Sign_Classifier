## File: preprocess _augmentation.py
## Name: Manuel Cuevas
## Date: 01/14/2017
## Project: CarND - LaneLines
## Desc: Augmentation pipeline; techniques like Random rotations, Zoom,
##       brightness, shear, translation and color tones are used here.
## Usage: Data augmentation allows the network to learn the important features
##      that are invariant for the object classes. In addition, augmentation is
##      used to artificially increase the size of the dataset
## This project was part of the CarND program.
## Tools learned in class were used to identify lane lines on the road.
## Revision: Rev 0000.005
## Split balance and augmentation functions
## Add shear, translation and color tones image augmentation
#######################################################################################
#importing useful packages
import numpy as np
import pickle
import cv2
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift
from skimage import exposure
import tqdm
import time
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


'''
Data augmentation will be used on a batch of images,
Thickness used random rotations, Zoom, Brightness, shear, translation, color tones, .
Input:  A list of images
Return: A list of augmented images'''
def augmentation(img, rotation = 20, shear_range = 10, trans_range = 8, bright_range = .4, color_range1 = .5, color_range2 = .1):
    rows,cols,ch = img.shape
    
    #Randomly rotate angle between -20° and 20° (uniform)
    degreeR = random.randint(-rotation, rotation)
    img = rotate(img, degreeR, reshape=False)
    
    # Translation ref- https://github.com/vxy10/ImageAugmentation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear ref- https://github.com/vxy10/ImageAugmentation
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    #Apply Translation+Shear
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    #Randomly change Brightness
    img = randomBrightness(img, bright_range)
    
    #change color tones
    img = randomColor(img, color_range1, color_range2)
    return img
            

'''
balance a data set of images to distribuite equal amount of classes
in addition images are augmented
Input:  image - Input image
Return: image with a small random color variation'''
def balance(images, labels, quantity):
    augmentedImgs = images
    
    with tqdm.tqdm(total=((len(images)+1)*quantity)) as pbar:
        for i in range (quantity):
            print (i)
            for idx, img in enumerate(augmentedImgs):
                #status bar
                pbar.update(1)

                #Augment image
                zm = augmentation(img,20,10,8,.4,0.2,.02)
                zm = zm.reshape(1, 32, 32, 3)
                
                #append augmented image
                images = np.concatenate((images, zm), axis=0)
                labels = np.append(labels, labels[idx])
            
    return images, labels

'''
Loads training data set, if data has not been previously balanced,
this function will balance the dataset to be equally distributed among
the classes, and then it saves it as preProcData_pickle. In addition this
function will call the augmentation function to increase the dataset as needed
Input:  None
Return: A list of balanced and augmented images'''
def pre_process_data(X_Train, y_Train, quantity = 9):
    #data before balance
    classes, counts = np.unique(y_Train, return_counts = True)
    print("Data before balance", counts)

    listItems = np.argwhere(y_Train == 0)
    indexList = []

    classes, counts = np.unique(y_Train, return_counts = True)
    for i in range (len(classes)):
        indexes = np.argwhere(y_Train == i)
        indexList.append(indexes[:min(counts)])

    ####create the new list
    deleteList = []
    for i in range (len(y_Train)):
        if(not(any(i in idx for idx in indexList))):
            deleteList.append(i)

    #Delete extra data
    y_Train = np.delete(y_Train, deleteList, None)
    X_Train = np.delete(X_Train, deleteList, 0)

    #Data after balance
    classes, counts = np.unique(y_Train, return_counts = True)
    print("Data After balance", counts)

    #Return: A list of augmented images
    X_Data, y_Data = balance(X_Train, y_Train, quantity)

    #Data after augmentation
    classes, counts = np.unique(y_Data, return_counts = True)
    print("Data After augmentation", counts)

    return X_Data, y_Data

'''
zooms in/out on an image
Input:  img - Input image
        zoom_factor - factor to zoom
Return: zoom image
http://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions'''
def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a couple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_factor, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_factor, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    else:
        out = img
    return out

'''
Randomly change image color tone
Input:  image - Input image
Return: image with a small random color variation'''
def randomColor(image, factor1 = 5, factor2 = 2):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    
    #color
    random_factor = np.random.uniform(1-factor1,1+factor1)
    image[:,:,0] = image[:,:,0]*random_factor
    random_factor = np.random.uniform(1-factor2,1+factor2)
    image[:,:,2] = image[:,:,2]*random_factor
    
    image = cv2.cvtColor(image,cv2.COLOR_HLS2RGB)
    return image

'''
Randomly change the brightness of the image
Input:  img - Input image
Return: image with a random brightness '''
def randomBrightness(img, b_range):
    gamma = np.random.uniform(1-b_range, 1+b_range)
    img = exposure.adjust_gamma(img, gamma, 1)
    return img

'''
Plots examples of augmented images
Input:  X_train - An array of images
Return: none '''
def visualize_augmentation(X_train):
    gs1 = gridspec.GridSpec(10, 10)
    gs1.update(wspace=0.01, hspace=0.02) # set the spacing between axes.
    plt.figure(figsize=(12,12))

    index = 3551
    image = X_train[3551]

    for i in range(0,10):
        for n in range(0,10):
            ax1 = plt.subplot(gs1[((i*10)+n)])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            img = augmentation(image,20,10,8,.4,0.2,.02)

            plt.subplot(10,10,((i*10)+n)+1)
            if (n > 0):
                plt.imshow(img)
            else:
                plt.imshow(image)

            plt.axis('off')
        index = random.randint(0, len(X_train))
        image = X_train[index]
    plt.show()


