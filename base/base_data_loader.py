## reference to : https://github.com/SantoshPattar/ConvNet-OOP/blob/master/base/data_loader_base.py

import numpy as np

class DataLoader(object):
    def __init__(self,config):
        self.config = config
        self.load_dataset()
#         self.calculate_class_label_size()
#         self.print_dataset_details()
#         self.preprocess_dataset()
        
    def load_dataset(self):
        """
        Loads the dataset.
        :param data_path optional
        :return none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to read the dataset from the disk.
        # Update respective data members of the DataLoader class.
        raise NotImplementedError
    
    def print_dataset_details(self):
        """
        Prints the details of the dataset (training & testing size).
        :param none
        :return none
        :raises none
        """
        return NotImplementedError
    
    def calculate_class_label_size(self):
        """
        Calculates the total number of classes in the dataset.
        :param none
        :return none
        """

        return NotImplementedError

    def display_data_element(self, which_data, index):
        """
        Displays a data element from a particular dataset (training/testing).
        :param which_data: Specifies the dataset to be used (i.e., training or testing).
        :param index: Specifies the index of the data element within a particular dataset.
        :returns none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to display a given data element.
        raise NotImplementedError

        
    def get_data_element(self, which_data, index):
        """
        Gets a data element from a particular dataset (training/testing).
        :param which_data: Specifies the dataset to be used (i.e., training or testing).
        :param index: Specifies the index of the data element within a particular dataset.
        :returns none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to display a given data element.
        raise NotImplementedError
        
    def preprocess_dataset(self):
        """
        Preprocess the dataset.
        :param none
        :returns none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to pre-process the dataset.

        raise NotImplementedError
        
    def shuffle_batch(self,batch_size):

        raise NotImplementedError
        
   
