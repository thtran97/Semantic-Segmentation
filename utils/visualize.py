import tensorflow as tf
import os
import matplotlib.pyplot as plt
from scipy.misc import *
import numpy as np


#Create a function to convert each pixel label to colour, given a color dict
def grayscale_to_colour(image,colordict):
    print('Converting image...')
    image = image.reshape((image.shape[0], image.shape[1], 1))
    image = np.repeat(image, 3, axis=-1)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = int(image[i][j][0])
            image[i][j] = np.array(colordict[label]) 
    return image

#Create a function to convert each pixel label to background colour, given a color dict
def grayscale_to_background(image,colordict):
    print('Converting background image...')
    image = image.reshape((image.shape[0], image.shape[1], 1))
    image = np.repeat(image, 4, axis=-1)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = int(image[i][j][0])
            image[i][j] = np.array(colordict[label] + [127]) 
    return image

def visualize_output(image,mask,colordict):
    mask = grayscale_to_background(mask,colordict)    
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    plt.imshow(street_im)
    plt.show()
#     imsave(photo_dir + "/image_%s.png" %(i*10 + j), converted_image)
    return street_im
