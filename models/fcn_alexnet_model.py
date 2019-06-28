from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models
import scipy
import numpy as np
import matplotlib.pyplot as plt

CHANNELS = 3
N_CLASSES = 2


class FcnAlexnetModel(BaseModel):
    def __init(self,config):
        super(FcnAlexnetModel,self).__init__(config)
        
    def build_model(self):
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

        self.deconv = tf.layers.conv2d_transpose(self.conv7,filters=N_CLASSES,
                                        kernel_size=63,
                                        strides = 32,
                                        padding = 'SAME',
                                        name="deconv")

        with tf.name_scope("output") : 
            self.logits  = tf.reshape(self.deconv,shape=(-1,N_CLASSES),name="logits")
            self.y_proba = tf.nn.softmax(self.logits,name="y_proba")

        with tf.name_scope('loss') : 
            self.y_flatten = tf.reshape(self.y,shape=[-1])
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.y_flatten,name="cross_entropy")
            self.loss_op = tf.reduce_mean(self.cross_entropy,name='fcn_loss')
                        
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): 
                self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate) 
                # training by optimizing the loss function
                # increment global_step by 1 after a training step
                self.training_step =self.optimizer.minimize(self.loss_op,global_step=self.global_step_tensor,name="training_op")
                
        with tf.name_scope('eval') : 
            self.correct = tf.nn.in_top_k(self.logits,self.y_flatten,1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct,tf.float32))

        
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    
#     def summary(self):
#         if self.model is None : 
#             print("You need to create a model first")
#         self.model.summary()

    def predict(self,im_input,im_output=None) :
        with tf.Session() as sess :
            self.load(sess)
            Z = sess.run(self.y_proba,feed_dict={self.X : [im_input],self.is_training:False})
            segmentation = np.argmax(Z,axis=1).reshape(self.height,self.width,1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(im_input)
            street_im.paste(mask, box=None, mask=mask)
            plt.imshow(street_im)
            plt.show()  

        
            
    
  

        