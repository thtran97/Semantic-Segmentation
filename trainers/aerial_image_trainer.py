from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf

class AerialImageTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(AerialImageTrainer, self).__init__(sess, model, data, config, logger)

    
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
        if test_loss < self.model.best_loss.eval(self.sess) :   
#             self.change_loss =  tf.assign(self.model.best_loss, test_loss)
            self.sess.run(self.model.change_loss,feed_dict={self.model.test_loss : test_loss})
            self.checks_without_progress = 0
            print("[BEST LOST : {}]".format(self.model.best_loss.eval(self.sess)))
            # save the model after each epoch
            self.model.save(self.sess)
        else:
            self.checks_without_progress += 1
            if self.checks_without_progress > self.max_checks_without_progress : 
                self.stop = True
                print("Early Stopping !")

        
    def train_step(self):
        
        X_batch,y_batch = next(self.data.shuffle_batch(self.config.batch_size))
        
        feed_dict = {self.model.X : X_batch, self.model.y : y_batch, self.model.is_training : True}
        
        _,loss,acc = self.sess.run([self.model.training_step,
                                   self.model.loss_op,
                                   self.model.accuracy],
                                   feed_dict = feed_dict)
                     
        return loss,acc
    
    
    def evaluate_model(self):
        losses = []
        accs = []
        for X_valid_batch,y_valid_batch in self.data.shuffle_batch(self.config.batch_size,"valid") :
            loss,acc = self.sess.run([self.model.loss_op,
                           self.model.accuracy],
                           feed_dict = {self.model.X : X_valid_batch, self.model.y : y_valid_batch, self.model.is_training : False })
            losses.append(loss)
            accs.append(acc)
        
        loss = np.mean(losses)
        acc = np.mean(accs)
        print("-->Last test loss      : ",loss)
        print("-->Last test accuracy  : ",acc)
        
        return loss,acc

 
        