from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras import layers,losses,models
import scipy
import numpy as np
import matplotlib.pyplot as plt
import sys,os 
import utils.losses as ul  
import utils.metrics as um


class Fcn8sModel(BaseModel):
    def __init(self,config):
        super(Fcn8sModel,self).__init__(config)

            
    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
    
#     def summary(self):
#         if self.model is None : 
#             print("You need to create a model first")
#         self.model.summary()

    
    def build(self,vgg16_npy_path=None):
        
        self.is_training = tf.placeholder(tf.bool)
        [self.height,self.width,self.channels] = self.config.image_size
        self.n_classes = 2
        with tf.name_scope("inputs") : 
            self.X = tf.placeholder(tf.float32,shape=(None,self.height,self.width,self.channels),name="X")
            self.y = tf.placeholder(tf.int32,shape=(None,self.height,self.width),name="y")
        
#         if vgg16_npy_path is None:
#             path = sys.modules[self.__class__.__module__].__file__
#             # print path
#             path = os.path.abspath(os.path.join(path, os.pardir,os.pardir))
#             # print(path)
#             path = os.path.join(path, "pretrained","vgg16.npy")
#             print(path)
#             vgg16_npy_path = path

#         self.data_dict = np.load(open(vgg16_npy_path,"rb"), encoding='latin1').item()
#         print("npy file loaded")
       
#         print(self.data_dict.keys())
        if vgg16_npy_path is None :
            self.conv1_1 =  tf.layers.conv2d(self.X, 64, kernel_size=3, 
                                         padding='same', 
                                         name="conv1_1")
            self.conv1_2 =  tf.layers.conv2d(self.conv1_1, 64, kernel_size=3, 
                                         padding='same', 
                                         name="conv1_2")
            self.pool1 = tf.nn.max_pool(self.conv1_2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name='pool1')
            
            
            self.conv2_1 =  tf.layers.conv2d(self.pool1, 128, kernel_size=3, 
                                         padding='same', 
                                         name="conv2_1")
            self.conv2_2 =  tf.layers.conv2d(self.conv2_1, 128, kernel_size=3, 
                                         padding='same', 
                                         name="conv2_2")
            self.pool2 = tf.nn.max_pool(self.conv2_2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name='pool2')
            
            self.conv3_1 =  tf.layers.conv2d(self.pool2, 256, kernel_size=3, 
                                         padding='same', 
                                         name="conv3_1")
            self.conv3_2 =  tf.layers.conv2d(self.conv3_1, 256, kernel_size=3, 
                                         padding='same', 
                                         name="conv3_2")
            self.conv3_3 =  tf.layers.conv2d(self.conv3_2, 256, kernel_size=3, 
                                         padding='same', 
                                         name="conv3_3")
            self.pool3 = tf.nn.max_pool(self.conv3_3,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name='pool3')
            
            self.conv4_1 =  tf.layers.conv2d(self.pool3, 512, kernel_size=3, 
                                         padding='same', 
                                         name="conv4_1")
            self.conv4_2 =  tf.layers.conv2d(self.conv4_1, 512, kernel_size=3, 
                                         padding='same', 
                                         name="conv4_2")
            self.conv4_3 =  tf.layers.conv2d(self.conv4_2, 512, kernel_size=3, 
                                         padding='same', 
                                         name="conv4_3")
            self.pool4 = tf.nn.max_pool(self.conv4_3,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name='pool4')
            
            
            self.conv5_1 =  tf.layers.conv2d(self.pool4, 512, kernel_size=3, 
                                         padding='same', 
                                         name="conv5_1")
            self.conv5_2 =  tf.layers.conv2d(self.conv5_1, 512, kernel_size=3, 
                                         padding='same', 
                                         name="conv5_2")
            self.conv5_3 =  tf.layers.conv2d(self.conv5_2, 512, kernel_size=3, 
                                         padding='same', 
                                         name="conv5_3")
            self.pool5 = tf.nn.max_pool(self.conv5_3,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name='pool5')
            
           
        else : 
            
            self.data_dict = np.load(open(vgg16_npy_path,"rb"), encoding='latin1').item()
            print("npy file loaded")

            print(self.data_dict.keys())

            self.conv1_1 = self._conv_layer(self.X, "conv1_1")
            self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self._max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self._max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self._max_pool(self.conv3_3, 'pool3')

            self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
            self.pool4 = self._max_pool(self.conv4_3, 'pool4')

            self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
            self.pool5 = self._max_pool(self.conv5_3, 'pool5')
            

        self.fc6 = tf.layers.conv2d(self.pool5,filters=4096,
                                 kernel_size=7,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="fc_6")

        self.drop6 = tf.layers.dropout(self.fc6,rate=0.5,name='dropout6')

        self.fc7 = tf.layers.conv2d(self.drop6,filters=4096,
                                 kernel_size=1,
                                 strides=1,
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 name="fc_7")

        self.drop7 = tf.layers.dropout(self.fc7,rate=0.5,name='dropout7')

        self.conv7_1x1 = tf.layers.conv2d(self.drop7, self.n_classes, kernel_size=1, 
                                         padding='same', 
                                         name="conv7_1x1")

        self.deconv7 =  tf.layers.conv2d_transpose(self.conv7_1x1, self.n_classes, 
                                                   kernel_size=4, 
                                                   strides=2, 
                                                   padding='same',
                                                   name = "deconv7")

        self.pool4_1x1 = tf.layers.conv2d(self.pool4, self.n_classes, kernel_size=1, 
                                         padding='same', 
                                         name="pool4_1x1")

        self.fuse1 = tf.add(self.deconv7,self.pool4_1x1, name = "fuse1")

        self.deconv_fuse1 = tf.layers.conv2d_transpose(self.fuse1, self.n_classes, 
                                                       kernel_size=4, 
                                                       strides=2, 
                                                       padding='same',
                                                       name = "deconv_fuse1")

        self.pool3_1x1 = tf.layers.conv2d(self.pool3, self.n_classes, kernel_size=1, 
                                         padding='same', 
                                         name="pool3_1x1")

        self.fuse2 = tf.add(self.deconv_fuse1,self.pool3_1x1,name="fuse2")

        self.logits = tf.layers.conv2d_transpose(self.fuse2,self.n_classes,
                                                 kernel_size=16,
                                                 strides=8,
                                                 padding="same",
                                                 name = "logits")

        with tf.name_scope("outputs") : 
            self.y_proba = tf.nn.sigmoid(self.logits,name="y_proba")
            self.output = tf.reduce_max(self.y_proba,axis=3,name='output')

        with tf.name_scope('loss') : 
            self.loss_op = tf.reduce_mean(ul.f_loss(tf.cast(self.y,tf.float32),self.output,self.config.loss),name="fcn_loss")
             
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if vgg16_npy_path is not None :
                scope_name = "fc((.)+)|conv7(.)+|deconv(.)+|pool(.)+|logits"
            else :
                scope_name = "(.)*"
                
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_name)    
            with tf.control_dependencies(update_ops): 
                self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate) 
                # training by optimizing the loss function
                # increment global_step by 1 after a training step
                self.training_step =self.optimizer.minimize(self.loss_op,var_list=self.train_vars,global_step=self.global_step_tensor,name="training_op")
                
        with tf.name_scope('eval') : 
            self.accuracy = tf.reduce_mean(um.f_accuracy(tf.cast(self.y,tf.float32),self.output,self.config.accuracy),name="accuracy")
    
        print("Model built successfully.")
        
    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)
    

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu


    def get_conv_filter(self, name):
        return tf.Variable(self.data_dict[name][0], name="filter")
        # use tf.constant() to prevent retraining it accidentally
#         return tf.constant(self.data_dict[name][0], name="filter")



    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")
        # use tf.constant() to prevent retraining it accidentally
#         return tf.constant(self.data_dict[name][1], name="biases")

        
   
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
       
        
        
        
        
        
        
        
        
        
        
        
        
        