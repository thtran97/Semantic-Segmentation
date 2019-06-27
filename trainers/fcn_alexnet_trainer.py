from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class FcnAlexnetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(FcnAlexnetTrainer, self).__init__(sess, model, data, config,logger)

    
    def train(self):
