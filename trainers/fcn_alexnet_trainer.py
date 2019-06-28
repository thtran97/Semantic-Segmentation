from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class FcnAlexnetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(FcnAlexnetTrainer, self).__init__(sess, model, data, config,logger)

    
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
        print(" Last epoch loss : ",batch_loss)
        print(" Last epoch accuracy :", batch_acc)
        # compute the loss and accruracy on test
        test_loss,test_acc = self.evaluate_model()

        # take the step number to write summary
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict ={
            'batch_loss' : batch_loss,
            'batch_accuracy' : batch_acc,
            'test_loss' : test_loss,
            'test_accuracy' : test_acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        
        # save the model after each epoch
        self.model.save(self.sess)
        
    def train_step(self):
        
        X_batch,y_batch = next(self.data.shuffle_batch(self.config.batch_size))
        
        feed_dict = {self.model.X : X_batch, self.model.y : y_batch, self.model.is_training : True}
        
        _,loss,acc = self.sess.run([self.model.training_step,
                                   self.model.loss_op,
                                   self.model.accuracy],
                                   feed_dict = feed_dict)
                     
        return loss,acc
    
    
    def evaluate_model(self):
        _,loss,acc = self.sess.run([self.model.training_step,
                           self.model.loss_op,
                           self.model.accuracy],
                           feed_dict = {self.model.X : self.data.test_data, self.model.y : self.data.test_mask, self.model.is_training : False })
        print("Last test loss :",loss)
        print("Last test accuracy : ",acc)
        return loss,acc
    

        