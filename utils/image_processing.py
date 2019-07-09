import numpy as np
from skimage.transform import resize


def build_colormap2label(colorlist) : 
    """Build a RGB color to label mapping for segmentation"""
    colormap2label = np.zeros(256**3)
    for i, colormap in enumerate(colorlist) : 
        # define the index of this color in map
        index  = colormap[0]*256**2 + colormap[1]*256 + colormap[2]
        # define the label of this color
        colormap2label[index] = i     
    return colormap2label

def label_indices(colormap,colormap2label) : 
    """Map a colormap to label"""
    colormap = colormap.astype('int32')
    idx = colormap [:,:,0]*256**2 + colormap[:,:,1]*256 + colormap[:,:,2]
    return colormap2label[idx]

def crop_image(img,x,y,crop_size) : 
    if (x+crop_size[0] < img.shape[0]) and (y+crop_size[1] < img.shape[1]) : 
        return  img[x:x+crop_size[0],y:y+crop_size[1]]
#     else :    
# #         print("Error on crop_size !")
     



def prepare_image(image, target_width = 320, target_height = 480, max_zoom = 0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height
        
    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)
    
    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    
    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1]

    # Let's also flip the image horizontally with 50% probability:
    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    # Now, let's resize the image to the target dimensions.
    # The resize function of scikit-image will automatically transform the image to floats ranging from 0.0 to 1.0
    image = resize(image, (target_width, target_height))
    
    # Finally, let's ensure that the colors are represented as 32-bit floats:
    return image.astype(np.float32)