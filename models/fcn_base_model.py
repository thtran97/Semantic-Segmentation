from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models
import scipy
import numpy as np
import matplotlib.pyplot as plt
import utils.losses as ul  
import utils.metrics as um
# from tensorflow_graph_in_jupyter import show_graph

class FcnBaseModel(BaseModel):
    def __init(self,config):
        super(FcnBaseModel,self).__init__(config)
    
            
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    
    def build(self):
        self.is_training = tf.placeholder(tf.bool)
        [self.height,self.width,self.channels] = self.config.image_size
        self.n_classes = self.config.n_classes
        
        with tf.name_scope("inputs") : 
            self.X = tf.placeholder(tf.float32,shape=(None,self.height,self.width,self.channels),name="X")
            self.y = tf.placeholder(tf.int32,shape=(None,self.height,self.width),name="y")
        
        # FCN architecture, but without Batch Normalisation implementation
        self.conv1_1 = tf.layers.conv2d(self.X,filters=32,
                                 kernel_size=5,
                                 strides=2,
                                 padding="VALID",
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 name="conv1_1")
        
        self.conv1_2 = tf.layers.conv2d(self.conv1_1,filters=32,
                         kernel_size=3,
                         strides=1,
                         padding="SAME",
                         activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                         name="conv1_2")

        self.pool1 = tf.nn.max_pool(self.conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name="pool_1")

        self.conv2_1 = tf.layers.conv2d(self.pool1,filters=64,
                                 kernel_size=3,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 name="conv2_1")
        
        self.conv2_2 = tf.layers.conv2d(self.conv2_1,filters=64,
                         kernel_size=3,
                         strides=1,
                         padding="SAME",
                         activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                         name="conv2_2")

        self.pool2 = tf.nn.max_pool(self.conv2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name="pool_2")

        self.conv3_1 = tf.layers.conv2d(self.pool2,filters=96,
                                 kernel_size=3,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 name="conv3_1")
        
        self.conv3_2 = tf.layers.conv2d(self.conv3_1,filters=96,
                         kernel_size=3,
                         strides=1,
                         padding="SAME",
                         activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                         name="conv3_2")

        self.pool3 = tf.nn.max_pool(self.conv3_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name="pool_3")
 
        self.conv4_1 = tf.layers.conv2d(self.pool3,filters=128,
                                 kernel_size=3,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                 name="conv4_1")
        
        self.conv4_2 = tf.layers.conv2d(self.conv4_1,filters=128,
                         kernel_size=3,
                         strides=1,
                         padding="SAME",
                         activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                         name="conv4_2")

        self.pool4 = tf.nn.max_pool(self.conv4_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name="pool_4")
        
        self.conv_score = tf.layers.conv2d(self.pool4,filters=self.n_classes,
                                           kernel_size=1,
                                           strides=1,
                                           padding="SAME",
                                           name="conv_score")

        
        # Implementation of Base FCN
        self.output = tf.layers.conv2d_transpose(self.conv_score,filters=self.n_classes,
                                kernel_size=64,
                                strides = 32,
                                padding = "SAME",
                                activation = tf.nn.softmax,
                                name="output")

        # Implementation of Skip Connection Network
        # ....
        
        # Implementation of MLP
        # ....
    
        with tf.name_scope('flatten'):
            self.out_flatten = tf.reshape(self.output,shape=(-1,self.height*self.width,self.n_classes),name="output_flatten")
            self.y_flatten = tf.reshape(self.y,shape=(-1,self.height*self.width),name="y_flatten")

        with tf.name_scope("loss") : 
#             self.loss_op = tf.reduce_mean(ul.f_loss(tf.cast(self.y,tf.float32),self.output,self.config.loss),name="fcn_loss")
            self.loss_op = tf.reduce_mean(ul.f_loss(self.y_flatten,self.out_flatten,self.config.loss),name="fcn_loss")

            starter_learning_rate = self.config.learning_rate
            self.learning_rate=tf.compat.v1.train.exponential_decay(starter_learning_rate, self.global_step_tensor,15000, 0.1,staircase=True)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): 
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
#                 self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,momentum=0.9)
                # training by optimizing the loss function
                # increment global_step by 1 after a training step
                self.training_step =self.optimizer.minimize(self.loss_op,global_step=self.global_step_tensor,name="training_op")
                
        with tf.name_scope("eval") : 
#             self.accuracy = tf.reduce_mean(um.f_accuracy(tf.cast(self.y,tf.float32),self.output,self.config.accuracy),name="accuracy")
            self.accuracy = tf.reduce_mean(um.f_accuracy(self.y_flatten,self.out_flatten,self.config.accuracy),name="accuracy")
            self.pred_flatten = tf.argmax(self.out_flatten,axis=2)
            self.pred = tf.reshape(self.pred_flatten,shape=(-1,self.height,self.width),name="pred")
            self.mean_iou,self.conf_mat = tf.metrics.mean_iou(self.y_flatten,self.pred_flatten,self.n_classes)

        print("Model built successfully.")
               
    def predict(self,sess,im_input,im_output=None) :
#         output_pred = sess.run(self.output,feed_dict={self.X : [im_input],self.is_training:False})
#         segmentation = (output_pred>0.5).reshape(self.height,self.width,1)
        output_pred = sess.run(self.pred,feed_dict={self.X : [im_input],self.is_training:False})
#         segmentation = (output_pred>0).reshape(self.height,self.width)
        
#         if im_input is None : 
#             plt.subplot(121)
#             plt.imshow(im_input)
#             plt.subplot(122)
#             plt.imshow(output_pred[0])
#         else :    
#             plt.subplot(131)
#             plt.imshow(im_input)
#             plt.subplot(132)
#             plt.imshow(im_output)
#             plt.subplot(133)
#             plt.imshow(output_pred[0])
            
#         plt.show()    
        
#         mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
#         mask = scipy.misc.toimage(mask, mode="RGBA")
#         street_im = scipy.misc.toimage(im_input)
#         street_im.paste(mask, box=None, mask=mask)
#         plt.imshow(street_im)
#         plt.show()

        if im_output is not None:
            acc = sess.run(self.accuracy,feed_dict={self.X : [im_input], self.y : [im_output], self.is_training:False})
            print("Accuracy : ",acc)
            return output_pred[0],acc
        
            
    
  

        