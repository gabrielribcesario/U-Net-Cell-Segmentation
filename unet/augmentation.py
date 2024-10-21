import tensorflow as tf

@tf.function
def gaussian_filter(size=3, sigma=1.):
    inv_sigmaSqd2 = 1. / (2. * sigma * sigma)
    uv = tf.linspace(-1., 1., size)
    uv *= uv
    uv = tf.reshape(uv, (-1,1)) + tf.reshape(uv, (1,-1))
    kernel = tf.math.exp(-uv * inv_sigmaSqd2) * 0.3183098861837907 * inv_sigmaSqd2
    return kernel / tf.math.reduce_sum(kernel)

@tf.function
def elastic_deformation(image, alpha=1., sigma=10., grid_size=3, order=0, fill_mode="mirror", fill_value=0.):
    """
    Elastic deformation as described by Sigmard, 2003.
    Currently it only accepts grayscale images.

    image - Input image.
    alpha - Displacement scaling factor.
    sigma - Gaussian kernel std dev.
    grid_size - Gaussian kernel size.
    order - Coordinate mapping interpolation (see tf.keras.ops.image.map_coordinates documentation).
    fill_mode - Boundary fill mode.
    fill_value - Fill value for fill_mode="constant".
    """
    # Image shape and gaussian kernel calculation.
    image = tf.squeeze(image)
    h, w = image.shape
    kernel = tf.reshape(gaussian_filter(grid_size, sigma), (grid_size, grid_size, 1, 1))
    # Pixel displacement vectors.
    dx = tf.squeeze(tf.nn.depthwise_conv2d(tf.random.uniform((1, h, w, 1), -1., 1.), kernel, (1,1,1,1), "SAME")) * alpha
    dy = tf.squeeze(tf.nn.depthwise_conv2d(tf.random.uniform((1, h, w, 1), -1., 1.), kernel, (1,1,1,1), "SAME")) * alpha
    # Displacement calculation.
    y, x = tf.meshgrid(tf.range(h, dtype=tf.float32), tf.range(w, dtype=tf.float32))
    return tf.expand_dims(tf.keras.ops.image.map_coordinates(image, 
                                                             [x + dx, y + dy], 
                                                             order=order, fill_mode=fill_mode, fill_value=fill_value), axis=-1)

@tf.function
def grid_deformation(image, steps_x, steps_y, 
                     grid_size=3, order=0, fill_mode="mirror", fill_value=0.):
    """
    distort_limits = (float, float), abs(float) <= 1.
    minval, maxval = distort_limits

    steps_x = tf.random.uniform((1 + grid_size,), 1. + minval, 1. + maxval) 
    steps_y = tf.random.uniform((1 + grid_size,), 1. + minval, 1. + maxval)
    """
    image = tf.squeeze(image)
    h, w = image.shape
    dx = w // grid_size
    dy = h // grid_size

    X, Y = tf.constant((), tf.float32), tf.constant((), tf.float32)
    aux_x, aux_y = 0., 0.
    x, y = 0, 0
    for i in range(1 + grid_size):
        begin_x = x
        begin_y = y
        end_x = tf.math.minimum(x + dx, w)
        end_y = tf.math.minimum(y + dy, h)

        curve_x = aux_x + dx * steps_x[i]
        curve_y = aux_y + dy * steps_y[i]
        X = tf.keras.ops.append(X, tf.linspace(aux_x, curve_x, end_x - begin_x))
        Y = tf.keras.ops.append(Y, tf.linspace(aux_y, curve_y, end_y - begin_y))

        aux_x = curve_x
        aux_y = curve_y
        x += dx
        y += dy
    u, v = tf.meshgrid(X, Y)
    return tf.expand_dims(tf.keras.ops.image.map_coordinates(image, [v, u], 
                                                             order=order, fill_mode=fill_mode, fill_value=fill_value), axis=-1)