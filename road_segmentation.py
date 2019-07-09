import tensorflow as tf

from data_loader.kitty_road_data_loader import KittyRoadLoader
from models.fcn_alexnet_model import FcnAlexnetModel
from trainers.fcn_alexnet_trainer import FcnAlexnetTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import matplotlib.pyplot as plt
import os 
import sys


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = KittyRoadLoader(config)
 
    # create an instance of the model you want
    model = FcnAlexnetModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = FcnAlexnetTrainer(sess, model, data, config, logger)
#     #load model if exists
#     model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
