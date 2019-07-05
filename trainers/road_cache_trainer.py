from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf

class RoadCacheTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(RoadCacheTrainer, self).__init__(sess, model, data, config, logger)

    def train(self):
        print("Train with cache layer")
        """
        This method requires a large memory ...
        This work is not finished yet !
        """
        self.X_hidden = []
        for i in tqdm(range(len(self.data.train_data))) : 
            X_batch = self.data.train_data[0]
            hidden_batch = self.sess.run(self.model.pool5,feed_dict={self.model.X : [X_batch], self.model.is_training:False})
            self.X_hidden.append(hidden_batch) 
#         self.hidden_cache_test = self.sess.run(self.model.pool5,feed_dict={self.model.X : self.data.valid_data,self.model.is_training:False})
        
        print("Export successfuly cache features")
        print(len(self.X_hidden))
        
#         self.checks_without_progress = 0
#         self.stop = False
#         num_epochs = self.config.num_epochs + self.model.cur_epoch_tensor.eval(self.sess)
           
#         for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), num_epochs, 1):
#             if self.stop : 
#                 break 
#             print("Epoch ",cur_epoch)
#             self.train_epoch()
#             self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        losses = []
        accs = []
        for i in tqdm(range(self.config.num_iter_per_epoch)): 
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        # compute the loss and accuracy of epoch
        batch_loss = np.mean(losses)
        batch_acc = np.mean(accs)
        print("-->Last epoch loss     : ", batch_loss)
        print("-->Last epoch accuracy : ", batch_acc)
        # compute the loss and accruracy on test
        test_loss,test_acc = self.evaluate_model()

        # take the step number to write summary
        cur_it = self.model.global_step_tensor.eval(self.sess)
        batch_summaries_dict ={
            'batch_loss' : batch_loss,
            'batch_accuracy' : batch_acc,
        }        
        test_summaries_dict ={
            'test_loss' : test_loss,
            'test_accuracy' : test_acc,
        }
        self.logger.summarize(cur_it, summarizer="train", summaries_dict=batch_summaries_dict)
        self.logger.summarize(cur_it, summarizer="test", summaries_dict=test_summaries_dict)
        
        # Early Stopping Algo
        if test_loss < self.best_loss.eval(self.sess) :   
            # save the model after each epoch
            self.model.save(self.sess)
            self.change_loss =  tf.assign(self.best_loss, test_loss)
            self.sess.run(self.change_loss)
            self.checks_without_progress = 0
            print("[BEST LOST : {}]".format(self.best_loss.eval(self.sess)))
        else:
            self.checks_without_progress += 1
            if self.checks_without_progress > self.max_checks_without_progress : 
                self.stop = True
                print("Early Stopping !")

        
    def train_step(self):
        
        X_hidden_batch,y_batch = next(self.data.shuffle_batch_with(self.X_hidden,self.data.train_mask,self.config.batch_size))
        
        feed_dict = {self.model.pool5 : X_hidden_batch, self.model.y : y_batch, self.model.is_training : True}
        
        _,loss,acc = self.sess.run([self.model.training_step,
                                   self.model.loss_op,
                                   self.model.accuracy],
                                   feed_dict = feed_dict)
                     
        return loss,acc
    
    
    def evaluate_model(self):
        loss,acc = self.sess.run([self.model.loss_op,
                           self.model.accuracy],
                           feed_dict = {self.model.pool5 : self.hidden_cache_test, self.model.y : self.data.valid_mask, self.model.is_training : False })
        print("-->Last test loss      : ",loss)
        print("-->Last test accuracy  : ",acc)
        return loss,acc
   
  
 
        