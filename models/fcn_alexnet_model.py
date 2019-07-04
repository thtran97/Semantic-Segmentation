from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models
import scipy
import numpy as np
import matplotlib.pyplot as plt
import utils.losses as ul  
import utils.metrics as um
# from tensorflow_graph_in_jupyter import show_graph

class FcnAlexnetModel(BaseModel):
    def __init(self,config):
        super(FcnAlexnetModel,self).__init__(config)
    
            
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    
#     def summary(self):
#         if self.model is None : 
#             print("You need to create a model first")
# #         self.model.summary()
#         show_graph(tf.get_default_graph())


    
    def build(self):
        self.is_training = tf.placeholder(tf.bool)
        [self.height,self.width,self.channels] = self.config.image_size
        self.n_classes = 2
        with tf.name_scope("inputs") : 
            self.X = tf.placeholder(tf.float32,shape=(None,self.height,self.width,self.channels),name="X")
            self.y = tf.placeholder(tf.int32,shape=(None,self.height,self.width),name="y")
        
        # FCN-Alexnet architecture
        self.conv0 = tf.layers.conv2d(self.X,filters=96,
                                 kernel_size=11,
                                 strides=4,
                                 padding="VALID",
                                 activation=tf.nn.relu,
                                 name="conv_0")

        self.pool0 = tf.nn.max_pool(self.conv0,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool_0")

        self.conv1 = tf.layers.conv2d(self.pool0,filters=256,
                                 kernel_size=5,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_1")

        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool_1")

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

        self.pool2 = tf.nn.max_pool(self.conv4,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool_2")


        self.conv5 =  tf.layers.conv2d(self.pool2,filters=4096,
                                 kernel_size=6,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_5")

        self.dropout0 = tf.layers.dropout(self.conv5,rate=0.5,name="dropout0")

        self.conv6 =  tf.layers.conv2d(self.dropout0,filters=4096,
                                 kernel_size=1,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="conv_6")

        self.dropout1 = tf.layers.dropout(self.conv6,rate=0.5,name="dropout1")

        self.conv7 = tf.layers.conv2d(self.dropout1,filters=self.n_classes,
                                kernel_size=1,
                                strides=1,
                                padding="VALID",
                                name="conv7")

        self.logits = tf.layers.conv2d_transpose(self.conv7,filters=self.n_classes,
                                        kernel_size=63,
                                        strides = 32,
                                        padding = "SAME",
                                        name="logits")

        with tf.name_scope("output") : 
            self.y_proba = tf.nn.sigmoid(self.logits,name="y_proba")
            self.output = tf.reduce_max(self.y_proba,axis=3,name="output")

        with tf.name_scope("loss") : 
            self.loss_op = tf.reduce_mean(ul.f_loss(tf.cast(self.y,tf.float32),self.output,self.config.loss),name="fcn_loss")
                        
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): 
                self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate) 
                # training by optimizing the loss function
                # increment global_step by 1 after a training step
                self.training_step =self.optimizer.minimize(self.loss_op,global_step=self.global_step_tensor,name="training_op")
                
        with tf.name_scope("eval") : 
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
        
            
    
  

        