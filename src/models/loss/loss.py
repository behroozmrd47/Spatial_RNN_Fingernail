import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import epsilon

from src.utils.utility_functions import binarize_images


def bce_dice_loss(y_true, y_pred):
    def dice_coefficient(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
        return numerator / (denominator + epsilon())

    return binary_crossentropy(y_true, y_pred) - tf.math.log(dice_coefficient(y_true, y_pred) + epsilon())


def dice_coefficient(y_true, y_pred):
    """
    To evaluate the dice coefficient.
    :param y_true: The ground truth labeled images.
    :param y_pred: The predicted labeled images.
    :return: the calculated Dice coefficient.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.squeeze(y_true, -1)
    y_pred = np.squeeze(y_pred, -1)
    y_pred = binarize_images(y_pred, threshold=0.3)
    intersection = (y_true * y_pred).sum()
    dice_score = (2. * intersection) / (y_true.sum() + y_pred.sum())
    return dice_score


def dice_metric(y_true, y_pred):
    """
    To evaluate the dice coefficient.
    :param y_true: The ground truth labeled images.
    :param y_pred: The predicted labeled images.
    :return: the calculated Dice coefficient.
    """
    y_true_3 = y_true[:, :, :, 0]
    y_pred_3 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    intersect = tf.math.multiply(y_true_3, y_pred_3)
    dice_score = (2. * tf.reduce_sum(intersect)) / (tf.reduce_sum(y_true_3) + tf.reduce_sum(y_pred_3) + epsilon())
    return dice_score
