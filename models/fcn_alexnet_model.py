from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models
import scipy
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


class FcnAlexnetModel(BaseModel):
    def __init(self,config):
        super(FcnAlexnetModel,self).__init__(config)
    
            
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    
#     def summary(self):
#         if self.model is None : 
#             print("You need to create a model first")
#         self.model.summary()

    
    def build(self):
        self.is_training = tf.placeholder(tf.bool)
        [self.height,self.width] = self.config.image_size
        with tf.name_scope("inputs") : 
            self.X = tf.placeholder(tf.float32,shape=(None,self.height,self.width,CHANNELS),name="X")
            self.y = tf.placeholder(tf.int32,shape=(None,self.height,self.width),name="y")
        
        # FCN-Alexnet architecture
        self.conv0 = tf.layers.conv2d(self.X,filters=96,
                                 kernel_size=11,
                                 strides=4,
                                 padding="VALID",
                                 activation=tf.nn.relu,
                                 name="conv_0")

        self.pool0 = tf.nn.max_pool(self.conv0,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name='pool_0')

        self.conv1 = tf.layers.conv2d(self.pool0,filters=256,
                                 kernel_size=5,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_1")

        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name='pool_1')

        self.conv2 = tf.layers.conv2d(self.pool1,filters=384,
                                 kernel_size=3,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_2")

        self.conv3 = tf.layers.conv2d(self.conv2,filters=384,
                                 kernel_size=3,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_3")

        self.conv4 = tf.layers.conv2d(self.conv3,filters=256,
                                 kernel_size=3,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_4")

        self.pool2 = tf.nn.max_pool(self.conv4,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name='pool_2')


        self.conv5 =  tf.layers.conv2d(self.pool2,filters=4096,
                                 kernel_size=6,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_5")

        self.dropout0 = tf.layers.dropout(self.conv5,rate=0.5,name='dropout0')

        self.conv6 =  tf.layers.conv2d(self.dropout0,filters=4096,
                                 kernel_size=1,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_6")

        self.dropout1 = tf.layers.dropout(self.conv6,rate=0.5,name='dropout1')

        self.conv7 = tf.layers.conv2d(self.dropout1,filters=N_CLASSES,
                                kernel_size=1,
                                strides=1,
                                padding="VALID",
                                name="conv7")

        self.logits = tf.layers.conv2d_transpose(self.conv7,filters=N_CLASSES,
                                        kernel_size=63,
                                        strides = 32,
                                        padding = 'SAME',
                                        name="logits")

        with tf.name_scope("output") : 
#             self.logits  = tf.reshape(self.deconv,shape=(-1,N_CLASSES),name="logits")
            self.y_proba = tf.nn.sigmoid(self.logits,name="y_proba")
#             self.output = np.argmax(self.logits,axis=1)
#             self.output = tf.reshape(self.output,shape=(-1,self.height,self.width),name='output')
            self.output = tf.reduce_max(self.y_proba,axis=3,name='output')

        with tf.name_scope('loss') : 
#             self.y_flatten = tf.reshape(self.y,shape=[-1])
#             self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.y_flatten,name="cross_entropy")
            self.cross_entropy = tf.keras.losses.binary_crossentropy(tf.cast(self.y,tf.float32),self.output)
            self.loss_op = tf.reduce_mean(self.cross_entropy,name='fcn_loss')
                        
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): 
                self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate) 
                # training by optimizing the loss function
                # increment global_step by 1 after a training step
                self.training_step =self.optimizer.minimize(self.loss_op,global_step=self.global_step_tensor,name="training_op")
                
        with tf.name_scope('eval') : 
#             self.correct = tf.nn.in_top_k(self.logits,self.y_flatten,1)
#             self.accuracy = tf.reduce_mean(tf.cast(self.correct,tf.float32),name="accuracy")
#             self.accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(self.y,self.output),name="accuracy")
            self.accuracy = tf.reduce_mean(dice_coeff(tf.cast(self.y,tf.float32),self.output),name="accuracy")

    
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
        
            
    
  

        