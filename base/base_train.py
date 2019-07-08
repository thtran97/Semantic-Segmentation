import tensorflow as tf
import numpy as np

class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data

        # run init
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        print("Variables initialized")
        
    def train(self):
        self.max_checks_without_progress = 5
        self.checks_without_progress = 0
        self.stop = False
        num_epochs = self.config.num_epochs + self.model.cur_epoch_tensor.eval(self.sess)
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), num_epochs, 1):
            if self.stop : 
                break 
            print("Epoch ",cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.train_epoch()
        
#         self.model.load(self.sess)
#         self.model.save_final_model(self.sess)
            
           
    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
        

