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
def elastic_deformation(image, alpha=1., sigma=10., kSize=17, auto_kSize=True, order=0, fill_mode="mirror", fill_value=0.):
    """
    Elastic deformation as described by Simard et al., 2003.
    Their paper is available at: https://cognitivemedium.com/assets/rmnist/Simard.pdf
    
    image - Input (grayscale) image.
    alpha - Displacement scaling factor.
    sigma - Gaussian kernel std dev.
    kSize - Gaussian kernel size.
    auto_kSize - Whether to compute the kernel size from the std dev.
    order - Coordinate mapping interpolation (see tf.keras.ops.image.map_coordinates documentation).
    fill_mode - Boundary fill mode.
    fill_value - Fill value for fill_mode="constant".
    """
    if auto_kSize:
        kSize = tf.cast(tf.math.ceil(1. + 2. * (1. + (sigma - 0.8) * 10. / 3.)), tf.int32)
    kSize = tf.math.maximum(kSize, 1)
    # Image shape and gaussian kernel calculation.
    image = tf.squeeze(image)
    h, w = image.shape
    # Pixel displacement vectors.
    dx = tf.random.uniform((1, h, w, 1), -1., 1.)
    dy = tf.random.uniform((1, h, w, 1), -1., 1.)
    if kSize > 1:
        kernel = tf.reshape(gaussian_filter(kSize, sigma), (kSize, kSize, 1, 1))
        dx = tf.nn.depthwise_conv2d(dx, kernel, (1,1,1,1), "SAME")
        dy = tf.nn.depthwise_conv2d(dy, kernel, (1,1,1,1), "SAME")
    dx = tf.squeeze(dx * alpha)
    dy = tf.squeeze(dy * alpha)
    # Displacement calculation.s
    x, y = tf.meshgrid(tf.range(w, dtype=tf.float32), tf.range(h, dtype=tf.float32))
    return tf.expand_dims(tf.keras.ops.image.map_coordinates(image, [y + dy, x + dx], order=order, fill_mode=fill_mode, fill_value=fill_value), axis=-1)

@tf.function
def grid_deformation(image, distort_limits=(-0.3, 0.3), grid_size=5, order=0, fill_mode="mirror", fill_value=0.):
    """
    """
    image = tf.squeeze(image)
    h, w = image.shape
    dx_grid = w // grid_size
    dy_grid = h // grid_size

    minval, maxval = distort_limits
    steps_x = tf.random.uniform((1 + grid_size,), 1. + minval, 1. + maxval) 
    steps_y = tf.random.uniform((1 + grid_size,), 1. + minval, 1. + maxval)

    dx, dy = tf.constant((), tf.float32), tf.constant((), tf.float32)
    aux_x, aux_y = 0., 0.
    x, y = 0, 0
    for i in range(1 + grid_size):
        begin_x = x
        begin_y = y
        end_x = tf.math.minimum(x + dx_grid, w)
        end_y = tf.math.minimum(y + dy_grid, h)

        curve_x = aux_x + dx_grid * steps_x[i]
        curve_y = aux_y + dy_grid * steps_y[i]
        dx = tf.keras.ops.append(dx, tf.linspace(aux_x, curve_x, end_x - begin_x))
        dy = tf.keras.ops.append(dy, tf.linspace(aux_y, curve_y, end_y - begin_y))

        aux_x = curve_x
        aux_y = curve_y
        x += dx_grid
        y += dy_grid
    dx, dy = tf.meshgrid(dx, dy)
    return tf.expand_dims(tf.keras.ops.image.map_coordinates(image, [dy, dx], order=order, fill_mode=fill_mode, fill_value=fill_value), axis=-1)