import tensorflow as tf

def dice_coeff(y_true,y_pred) : 
    smooth = 1
    # flatten 
    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def f_loss(y_true, y_pred,loss_name=None):
    if loss_name == "bce_dice_loss":
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    elif loss_name == "dice_loss" : 
        loss = dice_loss(y_true, y_pred)
    else : 
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return loss



