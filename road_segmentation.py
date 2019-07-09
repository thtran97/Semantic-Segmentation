import tensorflow as tf

from data_loader.kitti_road_data_loader import KittiRoadLoader
from models.unet_model import UNetModel
from trainers.road_trainer import RoadTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import matplotlib.pyplot as plt
import os 

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)


    # create your data generator
    data = KittiRoadLoader(config)
 
    # create an instance of the model you want
    model = UNetModel(config)
    model.build() 
     
    # create a builder for saving the model 
#     builder = tf.saved_model.builder.SavedModelBuilder(config.final_model_dir)
    
    # create tensorflow session
    sess = tf.Session()

    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = RoadTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()
    
    # save the final model
#     model.load(sess)
#     print("Saving the final model..")
#     builder.add_meta_graph_and_variables(sess,
#                                        [tf.saved_model.tag_constants.TRAINING],
#                                        signature_def_map=None,
#                                        assets_collection=None)
#     builder.save()
#     print("Final model saved")
    
    sess.close()


if __name__ == '__main__':
    main()
