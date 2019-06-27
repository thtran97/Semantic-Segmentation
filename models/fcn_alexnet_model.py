from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models

CHANNELS = 3
N_CLASSES = 2

class FcnAlexnetModel(BaseModel):
    def __init(self,config):
        super(FcnAlexnetModel,self).__init__(config)
        
    def build_model(self):
        """"
        Keras Functional API
        """"
        [height,width] = self.config.image_size
        self.inputs = layers.Input(shape=(height,width,CHANNELS))
        
        # the base net
        self.conv1 = layers.Conv2D(96,kernel_size=(11,11),strides=(4,4),padding='valid')(self.inputs)
        self.conv1 = layers.Activation('relu')(self.conv1)
        self.pool1 = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(self.conv1)
        self.norm1 = layers.BatchNormalization()(self.pool1)
        
        self.conv2 = layers.Conv2D(256,kernel_size=(5,5),strides=(1,1),padding='same')(self.norm1)
        self.conv2 = layers.Activation('relu')(self.conv2)
        self.pool2 = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(self.conv2)
        self.norm2 = layers.BatchNormalization()(self.pool2)
        
        self.conv3 = layers.Conv2D(384,kernel_size=(3,3),strides=(1,1),padding='same')(self.norm2)
        self.conv3 = layers.Activation('relu')(self.conv3)
        self.conv4 = layers.Conv2D(384,kernel_size=(3,3),strides=(1,1),padding='same')(self.conv3)
        self.conv4 = layers.Activation('relu')(self.conv4)
        self.conv5 = layers.Conv2D(256,kernel_size=(3,3),strides=(1,1),padding='same')(self.conv4)
        self.conv5 = layers.Activation('relu')(self.conv5)
        self.pool5 = layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(self.conv5)
        
        # fully convolutional net
        self.fc6 = layers.Conv2D(4096,kernel_size=(6,6),strides=(1,1),padding='same')(self.pool5)
        self.fc6 = layers.Activation('relu')(self.fc6)
        self.drop6 = layers.Dropout(rate=0.5)(self.fc6)
        
        self.fc7 = layers.Conv2D(4096,kernel_size=(1,1),strides=(1,1),padding='same')(self.drop6)
        self.fc7 = layers.Activation('relu')(self.fc7)
        self.drop7 = layers.Dropout(rate=0.5)(self.fc7)
        
        self.fc8 = layers.Conv2D(N_CLASSES,kernel_size=(1,1),strides=(1,1),padding='same')(self.drop7)
        self.last_layer = layers.Conv2DTranspose(N_CLASSES,(63,63),strides=(32,32),padding="same")(self.fc8)
        self.outputs = layers.Softmax()(self.last_layer)
        
        self.model = models.Model([self.inputs],[self.outputs],name='fcn_alexnet')                           
        print("Build model successfully")
        
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['acc'],
        )
        
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    
    def summary(self):
        if self.model is None : 
            print("You need to create a model first")
        self.model.summary()
        
  

        