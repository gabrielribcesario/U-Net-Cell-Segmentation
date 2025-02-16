from .seg_model import UNet
import tensorflow as tf

class UNetHelper:
    def __init__(self, 
                 strategy,
                 model_param,
                 loss_func=tf.keras.losses.sparse_categorical_crossentropy,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                 opt_schedule=None
                 ):
        """
        A helper class meant for providing a more convenient way
        of creating customized distributed training loops for the U-Net.
        """
        # Distributed training strategy.
        self.strategy = strategy
        # Objective function (minimization).
        self.loss_func = loss_func
        # Optimizer and learning rate scheduler.
        self.optimizer = optimizer
        self.opt_schedule = opt_schedule
        # Pixel labelling accuracy.
        self.train_sca = tf.keras.metrics.SparseCategoricalAccuracy(name='train_sca')
        self.val_sca = tf.keras.metrics.SparseCategoricalAccuracy(name='val_sca')
        # Model creation.
        self.model_param = model_param
        self.create_model(**self.model_param)

    def create_model(self, input_shape=(224, 224, 3), **kwargs):
        self.model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=input_shape), 
                                                 UNet(2, **kwargs)])
        self.checkpoint = tf.train.Checkpoint(self.model)
        self.checkpoint_dir = "./models/ckpt/"

    @tf.function
    def train_step(self, inputs):
        X, y, sw = inputs
        with tf.GradientTape() as tape:
            # Predicted labels.
            pred = self.model(X, training=True)
            # Unweighted loss.
            obj_loss = self.loss_func(y, pred)
            # Average weighted loss.
            avg_loss = tf.nn.compute_average_loss(obj_loss, sample_weight=sw)
            avg_loss /= tf.cast(tf.math.reduce_prod(tf.shape(obj_loss)[1:]), tf.float32) # image size scaling
            # Regularization loss (i.e. L1 and/or L2 regularization).
            reg_loss = self.model.losses
            if reg_loss: 
                avg_loss += tf.nn.scale_regularization_loss(tf.add_n(reg_loss))
        gradients = tape.gradient(avg_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return avg_loss, self.train_sca(y, pred)

    @tf.function
    def val_step(self, inputs):
        X, y = inputs
        # Predicted labels.
        pred = self.model(X, training=False)
        # Average loss calculation.
        obj_loss = self.loss_func(y, pred)
        avg_loss = tf.nn.compute_average_loss(obj_loss)
        avg_loss /= tf.cast(tf.math.reduce_prod(tf.shape(obj_loss)[1:]), tf.float32) # image size scaling
        return avg_loss, self.val_sca(y, pred)

    @tf.function
    def dist_train_step(self, inputs):
        out = self.strategy.run(self.train_step, args=[inputs])
        return self.strategy.reduce('sum', out[0], axis=None), self.strategy.reduce('mean', out[1], axis=None)

    @tf.function
    def dist_val_step(self, inputs):
        out = self.strategy.run(self.val_step, args=[inputs])
        return self.strategy.reduce('sum', out[0], axis=None), self.strategy.reduce('mean', out[1], axis=None)