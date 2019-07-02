"""
Implement the KittiRoadLoader class by inheriting the DataLoader class
Load data frol the Kitty Road dataset and preprocess it...
## Reference to : https://github.com/SantoshPattar/ConvNet-OOP
## Dataset link  : http://www.cvlibs.net/datasets/kitti/eval_road.php
"""
from base.base_data_loader import DataLoader
import numpy as np
import os 
from glob import glob
import matplotlib.pyplot as plt
from skimage import io
from sklearn.utils import shuffle

COLORMAP = [[255,0,0],[255,0,255]]
CLASSES = ['background','road']

def build_colormap2label() : 
    """Build a RGB color to label mapping for segmentation"""
    colormap2label = np.zeros(256**3)
    for i, colormap in enumerate(COLORMAP) : 
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

def filter_images(imgs,crop_size) : 
    res = ( [img[:crop_size[0],0:crop_size[1]] for img in imgs if (img.shape[0] >= crop_size[0] and img.shape[1] >= crop_size[1])]
            + [img[:crop_size[0],200:200+crop_size[1]] for img in imgs if (img.shape[0] >= crop_size[0] and img.shape[1] >= crop_size[1])] 
            + [img[:crop_size[0],400:400+crop_size[1]] for img in imgs if (img.shape[0] >= crop_size[0] and img.shape[1] >= crop_size[1])]
            + [img[:crop_size[0],img.shape[1]-crop_size[1]:img.shape[1]] for img in imgs if (img.shape[0] >= crop_size[0] and img.shape[1] >= crop_size[1])])
    return res

def data_augementation(X,y,ax) :
    new_X,new_y = [],[]
    for i in range(len(X)) : 
        new_X.append(np.flip(X[i],axis=ax))
        new_y.append(np.flip(y[i],axis=ax))
    X = np.concatenate((X,new_X),axis=0)
    y = np.concatenate((y,new_y),axis=0)
    return X,y

class KittiRoadLoader(DataLoader):
    
    def __init__(self,config):
        """
        Constructor to initialize the training and testing datasets for Kitty Road
        """
        super().__init__(config)
    
    def load_dataset(self):
        """
        Load the dataset from the disk location
        """
        data_path = self.config.data_path
        print("Yeah this is the path to dataset :",data_path)
        self.all_raw_images,self.all_raw_labels,self.all_raw_masks = self.read_data(data_path)
        print("Pre-processing the data...")
        self.preprocess_data()
        print("Size of images collection : ",self.all_images.shape)
        print("Size of masks collection : ",self.all_masks.shape)
        
        # Read the training data images and their masks
        self.train_data = self.all_images[0:200]
        self.train_mask = self.all_masks[0:200]
        
        # Read the valid data images and their masks
#         self.valid_data = 
#         self.valid_mask =
        
        # Read the test data images and their masks
        self.test_data = self.all_images[200:300]
        self.test_mask = self.all_masks[200:300]
        
    def read_data(self,data_path):
        image_paths = glob(os.path.join(data_path,'training','image_2','*.png'))
        label_paths = glob(os.path.join(data_path,'training','gt_image_2','*_road_*.png'))
        images = [io.imread(image_path) for image_path in image_paths]
        labels = [io.imread(label_path) for label_path in label_paths]  
        masks = [label_indices(item,build_colormap2label()) for item in labels]
        print("Size of all raw images : ", len(images), "samples with size ",images[0].shape)
        print("Size of all raw labels  : ", len(labels), "samples with size ",labels[0].shape)
        print("Size of all raw masks  : ", len(masks), "samples with size ",masks[0].shape)
        return images,labels,masks
        
    def display_data_element(self,which_data,index):
        plt.figure()
        if(which_data == "train_data"):
            plt.subplot(1,2,1)
            plt.imshow(self.train_data[index])
            plt.subplot(1,2,2)
            plt.imshow(self.train_mask[index])
#         elif(which_data == "valid_data"):
#             plt.subplot(1,2,1)
#             plt.imshow(self.valid_data[index])
#             plt.subplot(1,2,2)
#             plt.imshow(self.valid_mask[index])
        elif(which_data == "test_data"):
            plt.subplot(1,2,1)
            plt.imshow(self.test_data[index])
            plt.subplot(1,2,2)
            plt.imshow(self.test_mask[index])
        else: 
            print("[Error from display_data_element] : which_data parameter is invalid ! It can be train_data,valid_data or test_data.")
        plt.show()
        plt.close()
        
    def get_data_element(self,which_data,index):
        if(which_data == "train_data"):
            return self.train_data[index],self.train_mask[index]
#         elif(which_data == "valid_data"):
        elif(which_data == "test_data"):
            return self.test_data[index],self.test_mask[index]
        elif(which_data == "all_data"):
            return self.all_images[index],self.all_masks[index]
        else: 
            print("[Error from display_data_element] : which_data parameter is invalid ! It can be train_data,test_data or all_data.")
            
    def preprocess_data(self):
        # export some images with the input size by sliding from the raw images 
        self.all_images = filter_images(self.all_raw_images,self.config.image_size)
        self.all_masks  = filter_images(self.all_raw_masks ,self.config.image_size)

        # data augementation by flipping 
        self.all_images,self.all_masks = data_augementation(self.all_images,self.all_masks,1)
        
        # shuffle
        self.all_images,self.all_masks = shuffle(self.all_images,self.all_masks)
        
        # normalization 
        self.all_images = np.array(self.all_images).astype(np.float32) / 255.0
        self.all_masks  = np.array(self.all_masks).astype(np.int32)
        
    def shuffle_batch(self,batch_size) : 
        rnd_idx = np.random.permutation(len(self.train_data))
        n_batches = len(self.train_data) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = self.train_data[batch_idx], self.train_mask[batch_idx]
            yield X_batch, y_batch
        
        
            
    
        