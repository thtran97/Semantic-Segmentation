"""
Implement the ISPRSLoader class by inheriting the DataLoader class
Load data from the ISPRS dataset and preprocess it...
## Reference to : 
## Dataset link  : 
"""
from base.base_data_loader import DataLoader
import numpy as np
import os 
from glob import glob
import matplotlib.pyplot as plt
from skimage import io
from sklearn.utils import shuffle
import utils.image_processing as img_process
from PIL import Image

# List each RGB color value in the labels and the categories they label

COLORMAP = [[255,0,0],[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0]]

CLASSES = ['background', 'imprevious surfaces / road','building', 'low vegetaion', 'tree', 'car']


class ISPRSLoader(DataLoader):
    
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
        self.read_data(data_path)
        self.preprocess_data()
        
        
    def read_data(self,data_path):
        train_set = {"13", "17", "1", "21", "23", "26", "32", "37","3", "5", "7"}
        valid_set = {"11", "15", "28", "30", "34"}

        train_image_set = [os.path.join(data_path,"top","top_mosaic_09cm_area"+item+".tif") for item in train_set]
        valid_image_set = [os.path.join(data_path,"top","top_mosaic_09cm_area"+item+".tif") for item in valid_set]
        train_gt_set = [os.path.join(data_path,"gts_for_participants","top_mosaic_09cm_area"+item+".tif") for item in train_set]
        valid_gt_set = [os.path.join(data_path,"gts_for_participants","top_mosaic_09cm_area"+item+".tif") for item in valid_set]

        self.train_image = [np.array(Image.open(path)) for path in train_image_set]
        self.valid_image = [np.array(Image.open(path)) for path in valid_image_set]
        self.train_gt = [np.array(Image.open(path)) for path in train_gt_set]
        self.valid_gt = [np.array(Image.open(path)) for path in valid_gt_set]
        self.train_mask = [img_process.label_indices(item,img_process.build_colormap2label(COLORMAP)) for item in self.train_gt]
        self.valid_mask = [img_process.label_indices(item,img_process.build_colormap2label(COLORMAP)) for item in self.valid_gt]
    
        print("Training set : {} images, Validation set : {} images".format(len(train_set),len(valid_set)))

        
    def display_data_element(self,which_data,index):
        plt.figure()
        if(which_data == "train_data"):
            plt.subplot(1,2,1)
            plt.imshow(self.train_norm_image[index])
            plt.subplot(1,2,2)
            plt.imshow(self.train_norm_mask[index])
        elif(which_data == "valid_data"):
            plt.subplot(1,2,1)
            plt.imshow(self.valid_norm_image[index])
            plt.subplot(1,2,2)
            plt.imshow(self.valid_norm_mask[index])
        elif(which_data == "test_data"):
            plt.subplot(1,2,1)
            plt.imshow(self.test_image[index])
        else: 
            print("[Error from display_data_element] : which_data parameter is invalid ! It can be train_data,valid_data or test_data.")
        plt.show()
        plt.close()
        
    def get_data_element(self,which_data,index):
        if(which_data == "train_data"):
            return self.train_norm_image[index],self.train_norm_mask[index]
        elif(which_data == "valid_data"):
            return self.valid_norm_image[index],self.valid_norm_mask[index]
        elif(which_data == "test_data"):
            return self.test_image[index],self.test_mask[index]
#         elif(which_data == "all_data"):
#             return self.all_images[index],self.all_masks[index]
        else: 
            print("[Error from display_data_element] : which_data parameter is invalid ! It can be train_data,test_data or all_data.")
            
    def preprocess_data(self):
        
        self.train_norm_image, self.train_norm_mask = [], []
        for idx,image in enumerate(self.train_image) : 
            self.train_norm_image = self.train_norm_image + img_process.split_image(image,self.config.image_size) 
        for idx,mask in enumerate(self.train_mask) : 
            self.train_norm_mask = self.train_norm_mask + img_process.split_image(mask,self.config.image_size)
            
        self.valid_norm_image, self.valid_norm_mask = [], []
        for idx,image in enumerate(self.valid_image) : 
            self.valid_norm_image = self.valid_norm_image + img_process.split_image(image,self.config.image_size) 
        for idx,mask in enumerate(self.valid_mask) : 
            self.valid_norm_mask = self.valid_norm_mask + img_process.split_image(mask,self.config.image_size)

    
        # data augementation by flipping 
        new_images = [np.flip(img,axis=1) for img in self.train_norm_image] # + [np.flip(img,axis=0) for img in self.train_norm_image]
        self.train_norm_image = np.concatenate((self.train_norm_image,new_images),axis=0)
        new_masks = [np.flip(img,axis=1) for img in self.train_norm_mask] # + [np.flip(img,axis=0) for img in self.train_norm_mask]
        self.train_norm_mask = np.concatenate((self.train_norm_mask,new_masks),axis=0)
        
        # shuffle
        self.train_norm_image,self.train_norm_mask = shuffle(self.train_norm_image,self.train_norm_mask)
        self.valid_norm_image,self.valid_norm_mask = shuffle(self.valid_norm_image,self.valid_norm_mask)
        
        # normalization 
        self.train_norm_image = np.array(self.train_norm_image).astype(np.float32) / 255.0
        self.train_norm_mask  = np.array(self.train_norm_mask).astype(np.int32)
        self.valid_norm_image = np.array(self.valid_norm_image).astype(np.float32) / 255.0
        self.valid_norm_mask  = np.array(self.valid_norm_mask).astype(np.int32)
        
                    
        print("Size of normalized training images : ",self.train_norm_image.shape)
        print("Size of normalized validation images : ",self.valid_norm_image.shape)
        print("Size of normalized training masks : ",self.train_norm_mask.shape)
        print("Size of normalized validation masks : ",self.valid_norm_mask.shape)
        
    def shuffle_batch(self,batch_size,which_data="train") : 
        if which_data=="train":
            rnd_idx = np.random.permutation(len(self.train_norm_image))
            n_batches = len(self.train_norm_image) // batch_size
            for batch_idx in np.array_split(rnd_idx, n_batches):
                X_batch, y_batch = self.train_norm_image[batch_idx], self.train_norm_mask[batch_idx]
                yield X_batch, y_batch
        elif which_data=="valid" :
            rnd_idx = np.random.permutation(len(self.valid_norm_image))
            n_batches = len(self.valid_norm_image) // batch_size
            for batch_idx in np.array_split(rnd_idx, n_batches):
                X_batch, y_batch = self.valid_norm_image[batch_idx], self.valid_norm_mask[batch_idx]
                yield X_batch, y_batch
            
        
            
    