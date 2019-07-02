from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models
import numpy as np
import matplotlib.pyplot as plt

CHANNELS = 3
N_CLASSES = 2

def dice_coeff(y_true,y_pred) : 
    smooth = 1
    # flatten 
    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def f_loss(y_true, y_pred):
#     loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
#     loss = dice_loss(y_true, y_pred)
    return loss

class UNetModel(BaseModel):
    def __init(self,config):
        super(UNetModel,self).__init__(config)
    
            
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    
#     def summary(self):
#         if self.model is None : 
#             print("You need to create a model first")
#         self.model.summary()

    
    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        [self.height,self.width] = self.config.image_size
        with tf.name_scope("inputs") :   
            self.X = tf.placeholder(tf.float32,shape=(None,self.height,self.width,CHANNELS),name="X")
            self.y = tf.placeholder(tf.float32,shape=(None,self.height,self.width),name="y")
        #     y = tf.reshape(y,shape=[-1,height,width,1])

        self.conv1_0 = tf.layers.conv2d(self.X,filters=16,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv1_0")
        self.conv1_1 = tf.layers.conv2d(self.conv1_0,filters=16,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv1_1")


        self.pool1 = tf.nn.max_pool(self.conv1_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name='pool_1')
        self.pool1 = tf.layers.dropout(self.pool1,rate=0.1)

        self.conv2_0 = tf.layers.conv2d(self.pool1,filters=32,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv2_0")
        self.conv2_1 = tf.layers.conv2d(self.conv2_0,filters=32,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv2_1")

        self.pool2= tf.nn.max_pool(self.conv2_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name='pool_2')
        self.pool2 = tf.layers.dropout(self.pool2,rate=0.1)

        self.conv3_0 = tf.layers.conv2d(self.pool2,filters=64,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv3_0")
        self.conv3_1 = tf.layers.conv2d(self.conv3_0,filters=64,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv3_1")

        self.pool3 = tf.nn.max_pool(self.conv3_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name='pool_3')
        self.pool3 = tf.layers.dropout(self.pool3,rate=0.1)

        self.conv4_0 = tf.layers.conv2d(self.pool3,filters=128,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv4_0")
        self.conv4_1 = tf.layers.conv2d(self.conv4_0,filters=128,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv4_1")

        self.pool4 = tf.nn.max_pool(self.conv4_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name='pool_4')
        self.pool4 = tf.layers.dropout(self.pool4,rate=0.1)

        self.conv5_0 = tf.layers.conv2d(self.pool4,filters=256,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv5_0")
        self.conv5_1 = tf.layers.conv2d(self.conv5_0,filters=256,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv5_1")

        self.u6_0 = tf.layers.conv2d_transpose(self.conv5_1,filters=128,
                                       kernel_size = 2,
                                       strides = 2,
                                       padding = "SAME",
                                       name = "u6_0")
        self.u6 = tf.keras.layers.concatenate([self.u6_0,self.conv4_1])
        self.u6 = tf.layers.dropout(self.u6,rate=0.1)

        self.conv6_0 = tf.layers.conv2d(self.u6,filters=128,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv6_0")
        self.conv6_1 = tf.layers.conv2d(self.conv6_0,filters=128,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv6_1")

        self.u7_0 = tf.layers.conv2d_transpose(self.conv6_1,filters=64,
                                       kernel_size = 2,
                                       strides = 2,
                                       padding = "SAME",
                                       name = "u7_0")
        self.u7 = tf.keras.layers.concatenate([self.u7_0,self.conv3_1])
        self.u7 = tf.layers.dropout(self.u7,rate=0.1)

        self.conv7_0 = tf.layers.conv2d(self.u7,filters=64,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv7_0")
        self.conv7_1 = tf.layers.conv2d(self.conv7_0,filters=64,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv7_1")

        self.u8_0 = tf.layers.conv2d_transpose(self.conv7_1,filters=32,
                                       kernel_size = 2,
                                       strides = 2,
                                       padding = "SAME",
                                       name = "u8_0")
        self.u8 = tf.keras.layers.concatenate([self.u8_0,self.conv2_1])
        self.u8 = tf.layers.dropout(self.u8,rate=0.1)

        self.conv8_0 = tf.layers.conv2d(self.u8,filters=32,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv8_0")
        self.conv8_1 = tf.layers.conv2d(self.conv8_0,filters=32,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv8_1")

        self.u9_0 = tf.layers.conv2d_transpose(self.conv8_1,filters=16,
                                       kernel_size = 2,
                                       strides = 2,
                                       padding = "SAME",
                                       name = "u9_0")
        self.u9 = tf.keras.layers.concatenate([self.u9_0,self.conv1_1])
        self.u9 = tf.layers.dropout(self.u9,rate=0.1)

        self.conv9_0 = tf.layers.conv2d(self.u9,filters=16,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv9_0")
        self.conv9_1 = tf.layers.conv2d(self.conv9_0,filters=16,
                             kernel_size=3,
                             strides=1,
                             padding="SAME",
                             activation=tf.nn.relu,
                             name="conv9_1")

        self.output = tf.layers.conv2d(self.conv9_1,filters=1,
                              kernel_size = 1,
                              strides=1,
                              padding = "SAME",
                              activation = tf.nn.sigmoid)

        self.output = tf.reshape(self.output,shape = (-1,self.height,self.width),name="output")

        with tf.name_scope('train') : 
#             loss_op = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y,output),name='fcn_loss')
#             self.loss_op = tf.reduce_mean(f_loss(self.y,self.output),name='fcn_loss')
            self.cross_entropy = tf.keras.losses.binary_crossentropy(tf.cast(self.y,tf.float32),self.output)           
            self.loss_op = tf.reduce_mean(self.cross_entropy,name='fcn_loss')
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): 
                self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                # training by optimizing the loss function
                # increment global_step by 1 after a training step
                self.training_step =self.optimizer.minimize(self.loss_op,global_step=self.global_step_tensor,name="training_op")

        with tf.name_scope('eval') : 
#             correct = tf.nn.in_top_k(logits,y_flatten,1)
#             accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
#             self.accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(self.y,self.output),name="accuracy")
            self.accuracy = tf.reduce_mean(dice_coeff(self.y,self.output),name="accuracy")






