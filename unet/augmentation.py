import tensorflow as tf

@tf.function
def gaussian_kernel(kSize=3, sigma=1.0):
    """
    Creates a 2D gaussian kernel using tensorflow.
    """
    inv_sigmaSqd2 = 1.0 / (2.0 * sigma * sigma)
    uv = tf.linspace(-1.0, 1.0, kSize)
    uv *= uv
    uv = tf.reshape(uv, (-1, 1)) + tf.reshape(uv, (1, -1))
    kernel = tf.math.exp(-uv * inv_sigmaSqd2) * 0.3183098861837907 * inv_sigmaSqd2
    return kernel / tf.math.reduce_sum(kernel)

@tf.function
def elastic_deformation(batch, alpha=1.0, sigma=10.0, kSize=17, auto_kSize=True, order=0, fill_mode="mirror", fill_value=0.0):
    """
    Elastic deformation as described by Simard et al., 2003. Available at: https://ieeexplore.ieee.org/document/1227801
    
    batch - Batch of images.
    alpha - Displacement scaling factor.
    sigma - Gaussian kernel std dev.
    kSize - Gaussian kernel size.
    auto_kSize - Whether to compute the kernel size from the std dev.
    order - Coordinate mapping interpolation (see tf.keras.ops.image.map_coordinates documentation).
    fill_mode - Boundary fill mode.
    fill_value - Fill value for fill_mode="constant".
    """
    # (batch size, height, width, number of channels)
    batch_shape = tf.shape(batch)
    # OpenCV formula.
    if auto_kSize or kSize == 0:
        kSize = tf.cast(tf.math.ceil(1.0 + 2.0 * (1.0 + (sigma - 0.8) * 10.0 / 3.0)), tf.int32) 
    kSize = tf.math.maximum(kSize, 1)
    # Pixel displacement vectors.
    dx = tf.random.uniform((1, batch_shape[1], batch_shape[2], 1), -1.0, 1.0)
    dy = tf.random.uniform((1, batch_shape[1], batch_shape[2], 1), -1.0, 1.0)
    if kSize > 1:
        kernel = tf.reshape(gaussian_kernel(kSize, sigma), (kSize, kSize, 1, 1))
        dx = tf.nn.depthwise_conv2d(dx, kernel, (1, 1, 1, 1), "SAME")
        dy = tf.nn.depthwise_conv2d(dy, kernel, (1, 1, 1, 1), "SAME")
    dx = tf.squeeze(dx * alpha)
    dy = tf.squeeze(dy * alpha)
    # Displacement calculation.
    y, x = tf.meshgrid(tf.range(batch_shape[1], dtype=tf.float32), tf.range(batch_shape[2], dtype=tf.float32))
    output = tf.TensorArray(batch.dtype, size=batch_shape[0])
    # Iterate batch
    for i in tf.range(batch_shape[0]):
        output_i = tf.TensorArray(batch.dtype, size=batch_shape[3])
        # Iterate channels
        for j in tf.range(batch_shape[3]):
            output_i = output_i.write(j, tf.keras.ops.image.map_coordinates(batch[i, ..., j], [y + dy, x + dx], 
                                                                            order=order, fill_mode=fill_mode, fill_value=fill_value))
        output_i = output_i.stack()
        output = output.write(i, tf.keras.ops.moveaxis(output_i, 0, -1))
    output = output.stack()
    return tf.reshape(output, batch_shape)

@tf.function
def grid_deformation(batch, distort_limits=(-0.3, 0.3), grid_size=5, order=0, fill_mode="mirror", fill_value=0.0):
    """
    A tensorflow implementation of the GridDistortion algorithm from Albumentations.
    See: https://github.com/albumentations-team/albumentations/blob/main/albumentations/augmentations/geometric/transforms.py
    """
    # (batch size, height, width, number of channels)
    batch_shape = tf.shape(batch)
    dx_grid = tf.cast(batch_shape[2] // grid_size, tf.float32)
    dy_grid = tf.cast(batch_shape[1] // grid_size, tf.float32)
    # Displacement grid.
    steps_x = tf.random.uniform((1 + grid_size,), 1.0 + distort_limits[0], 1.0 + distort_limits[1]) 
    steps_y = tf.random.uniform((1 + grid_size,), 1.0 + distort_limits[0], 1.0 + distort_limits[1])
    # Grid displacement vectors.
    dx = tf.TensorArray(tf.float32, size=1 + grid_size, infer_shape=False)
    dy = tf.TensorArray(tf.float32, size=1 + grid_size, infer_shape=False)
    aux_x, aux_y = 0.0, 0.0
    x, y = 0, 0
    for i in tf.range(1 + grid_size):
        begin_x = x
        begin_y = y
        end_x = tf.math.minimum(x + tf.cast(dx_grid, tf.int32), batch_shape[2])
        end_y = tf.math.minimum(y + tf.cast(dy_grid, tf.int32), batch_shape[1])

        curve_x = aux_x + dx_grid * steps_x[i]
        curve_y = aux_y + dy_grid * steps_y[i]
        dx = dx.write(i, tf.linspace(aux_x, curve_x, end_x - begin_x))
        dy = dy.write(i, tf.linspace(aux_y, curve_y, end_y - begin_y))

        aux_x = curve_x
        aux_y = curve_y
        x += tf.cast(dx_grid, tf.int32)
        y += tf.cast(dy_grid, tf.int32)
    dx = dx.concat()
    dy = dy.concat()
    dx, dy = tf.meshgrid(dx, dy)
    output = tf.TensorArray(batch.dtype, size=batch_shape[0])
    # Iterate batch
    for i in tf.range(batch_shape[0]):
        output_i = tf.TensorArray(batch.dtype, size=batch_shape[3])
        # Iterate channels
        for j in tf.range(batch_shape[3]):
            output_i = output_i.write(j, tf.keras.ops.image.map_coordinates(batch[i, ..., j], [dy, dx], 
                                                                            order=order, fill_mode=fill_mode, fill_value=fill_value))
        output_i = output_i.stack()
        output = output.write(i, tf.keras.ops.moveaxis(output_i, 0, -1))
    output = output.stack()
    return tf.reshape(output, batch_shape)