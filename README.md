# Sematic Segmentation Study



## About this project



This git is my study on Semantic Segmentation tasks with TensorFlow and Keras 



The template is forked from [TensorFlow Project Template](https://github.com/MrGemy95/Tensorflow-Project-Template)



## Release



* [Version 1.0](https://github.com/kuro10/Sematic-Segmentation/tree/6c3bab3126619621b238895e1f9a6f11563874cf) : The first version of this project is well done for training a FCN-Alexnet model with Kitti Road Dataset. However, it is quite difficult to restore this model for a prediction. So these constraints will be optimized in the following update. 

* [Version 1.1](https://github.com/kuro10/Semantic-Segmentation/tree/928f7b5b16fff1d7502bfe365353dd5c761abffe) : This version is the implementation of FCN-Alexnet, U-Net and FCN-8s (based on VGG16-Net), training with Kitti Road Dataset. Also, transfer learning and fine tuning are applied when using FCN-8s (restore weights from file (vgg16.npy)[https://github.com/machrisaa/tensorflow-vgg], another solution that is using the file (vgg_16.ckpt)[https://github.com/tensorflow/models/tree/master/research/slim] that will be developped in the future). The idea about "caching the frozen layers" is tested but not really work yet, this idea will be also developped in the future.  


## Instructions

### Training step

* Create a configuration file .json 

```
"exp_name" : Folder that you want save the checkpoints and summaries for tensorboard

"num_epochs" : Number of epochs for training, pay attention to overfitting !

"num_iter_per_epoch" : Number of iterations executed in each epoch

"learning_rate" : Used for optimizer. So, what is the best rate ? How to choose the best 
learning rate ?  

"batch_size" : Number of samples used for training in each iteration

"max_to_keep" :  Number of checkpoints maximum that you want to keep

"data_path" : Path to dataset

"image_size" : Input image size with format [height,width,channels]

"loss" : Name of loss function you want to use

"accuracy" : Name of accuracy function you want to use
```

* Read the config file

```python
from utils.config import process_config 

config = process_config("PATH/TO/CONFIG/FILE")
```

* Create a session 

```python
from tensorflow as tf

sess = tf.Session()
```

* Create your data generator

```python
from data_loader.kitti_road_data_loader import KittyRoadLoader

data = KittyRoadLoader(config)
```

* Create and build an instance of model

```python
from models.fcn_alexnet_model import FcnAlexnetModel

model  = FcnAlexnetModel(config)

model.build()
```

* Create an instance of logger for saving checkpoints and summaries.

```python
from utils.logger import Logger 

logger = Logger(sess,config)
```

* Create an trainer for training the created model with your above dataset

```python
from trainers.road_trainer import RoadTrainer

trainer = RoadTrainer(sess,model,data,config,logger)
```

* Load your model if exists

```python
model.load(sess)
```

* Train your model by the trainer

```python
trainer.train()
```

* Close the session when you finish 
```python
sess.close()
```
