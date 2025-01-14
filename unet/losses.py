from scipy.ndimage import distance_transform_edt
from skimage.measure import label
import tensorflow as tf
import numpy as np

def unet_sample_weights(x, w0=10., sigma=5., data_type=np.float32, use_distance_factor=True):
    """
    Given a map of pixel labels x ∈ omega, this function returns the weight map 
    w(x) = w_c(x) + w0 * exp(-(d1(x) + d2(x))^2/(2 * sigma^2)).
    
    Setting "use_distance_factor" to False has the same effect as setting w0 = 0, but results in fewer calculations.

    For more details: https://arxiv.org/pdf/1505.04597
    """
    if x.dtype != np.int32: 
        x = x.astype(np.int32)
    sample_weights = np.zeros(shape=x.shape, dtype=data_type)
    img_shape = x.shape
    img_size = np.prod(img_shape)
    # Calculates the class weights, least frequent classes get larger weights. I'm also normalizing 
    # the sample weights (sw) so that the most common label gets a sw = 1. and the least frequent ones get a sw > 1.
    # The following 2 lines of code can be used for calculating the class frequency for n_classes, btw.
    unique, idx = np.unique(x.ravel(), return_inverse=True)
    freq_classes = np.bincount(idx) / img_size
    w_c = freq_classes[idx].max() / freq_classes[idx].reshape(img_shape)
    sample_weights += w_c
    # Calculates the distance factor given the two smallest distances between two foreground objects.
    if use_distance_factor:
        # Computes the foreground objects' labels.
        objects_labels, num_labels = label(x, 0, True, 2)
        foreground_labels = np.unique(objects_labels)[1:]
        if num_labels >= 3: # 2 foreground objects (i.e. 3 labels) are required
            maps = np.zeros(shape=(num_labels - 1, img_shape[0], img_shape[1]), dtype=data_type)
            for j, map_j in enumerate(maps):
                map_j += distance_transform_edt(objects_labels != foreground_labels[j])
            # Sorts the 2 maps with the two smallest distances between two objects.
            d1, d2 = np.sort(maps, axis=0)[:2]
            sample_weights += w0 * np.exp(-(d1 + d2)**2. / (2. * sigma**2.))
    return sample_weights

@tf.function
def dice_loss(y_true, y_pred):
    ohe_y = tf.expand_dims(y_true, axis=-1) # one-hot encoded y_true
    ohe_y = tf.cast(tf.concat([1 - ohe_y, ohe_y], axis=-1), y_pred.dtype)
    dice =  1.0 - 2.0 * tf.reduce_sum(y_pred * ohe_y, axis=(1,2,3)
                                      ) / (tf.reduce_sum(ohe_y, axis=(1,2,3)) + tf.math.reduce_sum(y_pred, axis=(1,2,3)) + tf.keras.backend.epsilon())
    return dice
    
@tf.function
def IoU(y_true, y_pred):
    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.bool)
    y_true = tf.cast(y_true, tf.bool)
    tp = tf.math.reduce_sum(tf.cast(tf.math.logical_and(y_true, y_pred), tf.float32), axis=(1,2)) 
    fp = tf.math.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true), y_pred), tf.float32), axis=(1,2))
    fn = tf.math.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_pred), y_true), tf.float32), axis=(1,2))
    return tp / (tp + fp + fn + tf.keras.backend.epsilon())