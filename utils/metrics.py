import tensorflow as tf

def dice_coeff(y_true,y_pred) : 
    smooth = 1
    # flatten 
    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def f_accuracy(y_true,y_pred,acc_name=None):
    if acc_name == "dice_coeff" :
        acc = dice_coeff(y_true,y_pred)
    elif acc_name == "binary" :
        acc = tf.keras.metrics.binary_accuracy(y_true,y_pred)
#     elif acc_name ==  "sparse_categorical":
    else:
        acc = tf.keras.metrics.sparse_categorical_accuracy(y_true,y_pred)
    return acc