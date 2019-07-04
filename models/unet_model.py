from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models
import numpy as np
import matplotlib.pyplot as plt
import scipy
import utils.losses as ul  
import utils.metrics as um



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

    
    def build(self):
        self.is_training = tf.placeholder(tf.bool)
        [self.height,self.width,self.channels] = self.config.image_size
        self.n_classes = 2
        with tf.name_scope("inputs") : 
            self.X = tf.placeholder(tf.float32,shape=(None,self.height,self.width,self.channels),name="X")
            self.y = tf.placeholder(tf.int32,shape=(None,self.height,self.width),name="y")
        
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
            self.loss_op = tf.reduce_mean(ul.f_loss(tf.cast(self.y,tf.float32),self.output,self.config.loss),name="fcn_loss")
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): 
                self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                # training by optimizing the loss function
                # increment global_step by 1 after a training step
                self.training_step =self.optimizer.minimize(self.loss_op,global_step=self.global_step_tensor,name="training_op")

        with tf.name_scope('eval') : 
            self.accuracy = tf.reduce_mean(um.f_accuracy(tf.cast(self.y,tf.float32),self.output,self.config.accuracy),name="accuracy")
    
    def predict(self,sess,im_input,im_output=None) :
        output_pred = sess.run(self.output,feed_dict={self.X : [im_input],self.is_training:False})
        segmentation = (output_pred>0.5).reshape(self.height,self.width,1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(im_input)
        street_im.paste(mask, box=None, mask=mask)
        plt.imshow(street_im)
        plt.show()
        if im_output is not None:
            acc = sess.run(self.accuracy,feed_dict={self.X : [im_input], self.y : [im_output], self.is_training:False})
            print("Accuracy : ",acc)
            return acc





