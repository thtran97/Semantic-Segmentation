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
import utils.image_processing as img_process

COLORMAP = [[255,0,0],[255,0,255]]
CLASSES = ['background','road']




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
        self.train_data = self.all_images[:200]
        self.train_mask = self.all_masks[:200]
        
        # Read the valid data images and their masks
        self.valid_data = self.all_images[200:300] 
        self.valid_mask = self.all_masks[200:300] 
        
        # Read the test data images and their masks
        self.test_data = self.all_images[300:]
        self.test_mask = self.all_masks[300:]
        
        # free some unused vars
        self.all_images, self.all_masks, self.all_raw_images, self.all_raw_masks, self.all_raw_labels = [],[],[],[],[]
        
    def read_data(self,data_path):
        image_paths = glob(os.path.join(data_path,'training','image_2','*.png'))
        label_paths = glob(os.path.join(data_path,'training','gt_image_2','*_road_*.png'))
        images = [io.imread(image_path) for image_path in image_paths]
        labels = [io.imread(label_path) for label_path in label_paths]  
        masks = [img_process.label_indices(item,img_process.build_colormap2label(COLORMAP)) for item in labels]
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
        elif(which_data == "valid_data"):
            plt.subplot(1,2,1)
            plt.imshow(self.valid_data[index])
            plt.subplot(1,2,2)
            plt.imshow(self.valid_mask[index])
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
        elif(which_data == "valid_data"):
            return self.valid_data[index],self.valid_mask[index]
        elif(which_data == "test_data"):
            return self.test_data[index],self.test_mask[index]
#         elif(which_data == "all_data"):
#             return self.all_images[index],self.all_masks[index]
        else: 
            print("[Error from display_data_element] : which_data parameter is invalid ! It can be train_data,test_data or all_data.")
            
    def preprocess_data(self):
        # export some images with the input size by sliding from the raw images 
        self.all_images = [img_process.crop_image(img,0,0,self.config.image_size) for img in self.all_raw_images]
        self.all_masks = [img_process.crop_image(img,0,0,self.config.image_size) for img in self.all_raw_masks]
    
        # data augementation by flipping 
        new_images = [np.flip(img,axis=1) for img in self.all_images]
        self.all_images = np.concatenate((self.all_images,new_images),axis=0)
        new_masks = [np.flip(img,axis=1) for img in self.all_masks]
        self.all_masks = np.concatenate((self.all_masks,new_masks),axis=0)
        
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
            
    def shuffle_batch_with(self,X_train,y_train,batch_size):
        rnd_idx = np.random.permutation(len(X_train))
        n_batches = len(X_train) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
            yield X_batch, y_batch
        
            
    
        