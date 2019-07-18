import numpy as np
from skimage.transform import resize


def build_colormap2label(colordict) : 
    """Build a RGB color to label mapping for segmentation"""
    colormap2label = np.zeros(256**3)
    for i, colormap in enumerate(colordict) : 
        # define the index of this color in map
        index  = colormap[0]*256**2 + colormap[1]*256 + colormap[2]
        # define the label of this color
#         colormap2label[index] = i     
        if i == 1 : #road
            colormap2label[index] = 1
        elif i == 2 : #building
            colormap2label[index] = 2
        else : 
            colormap2label[index] = 0     
    return colormap2label

def label_indices(colormap,colormap2label) : 
    """Map a colormap to label"""
    colormap = colormap.astype('int32')
    idx = colormap[:,:,0]*256**2 + colormap[:,:,1]*256 + colormap[:,:,2]
    return colormap2label[idx]

    
def crop_image(img,x,y,crop_size) : 
    if (x+crop_size[0] < img.shape[0]) and (y+crop_size[1] < img.shape[1]) : 
        return  img[x:x+crop_size[0],y:y+crop_size[1]]
#     else :    
# #         print("Error on crop_size !")
     
def split_image(img,split_size):
    res = []
    for x in range(0,img.shape[0],split_size[0]) : 
        for y in range(0,img.shape[1],split_size[1]):
            if x+split_size[0] <= img.shape[0] and y+split_size[1] <= img.shape[1] : 
                subimg = img[x:x+split_size[0],y:y+split_size[1]]
                res.append(subimg)
    return res

def merge_image(img,img_list,split_size):
    new_img = np.zeros(img.shape)
    for x in range(0,img.shape[0],split_size[0]) : 
        for y in range(0,img.shape[1],split_size[1]):
            if x+split_size[0] <= img.shape[0] and y+split_size[1] <= img.shape[1] : 
                new_img[x:x+split_size[0],y:y+split_size[1]] = img_list.pop(0)
    return new_img