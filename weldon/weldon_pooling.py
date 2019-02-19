"""
Implementation of WeldonPooling2dLayer.

Inspired by https://stackoverflow.com/questions/49988402/has-anyone-written-weldon-pooling-for-keras
"""

# Standard imports

# External librairies imports
import tensorflow as tf
from keras.layers import Layer

# Internal imports


# During backpropagation MinMax only let the gradient through the selected tiles
@tf.custom_gradient
def weldon_pooling_2d(inputs, kmin, kmax):
    """
    Define the Weldon Pooling 2D Operation.

    Parameters
    ----------
    inputs: tf.Tensor (BATCH_SIZE x N x M x N_CHANNELS)
        inputs of the layer during the forward propagation.
    kmin: int
        number of minimal values to retain.
    kmax: int
        number of maximal values to retain.

    Return
    ------
    outputs: tf.Tensor (BATCH_SIZE x N_CHANNELS)
        outputs of the layer during the forward propagation.
    grad : function
        gradients' function of the operation according param:inputs, param:kmin, param:kmax.
    """
    # Initialization
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    n_channels = input_shape[1]
    h = input_shape[2]
    w = input_shape[3]
    n = h * w

    inputs = tf.reshape(inputs, [batch_size, n_channels, n])

    # Sort in descending order
    sorted_inputs, indices = tf.nn.top_k(inputs, k=n, sorted=True) # descending order

    # Compute outputs
    outputs = tf.reduce_mean(sorted_inputs[:, :, :kmax], axis=2)

    outputs = tf.cond(tf.greater(kmin, 0),
                      lambda: tf.add(outputs, tf.reduce_mean(sorted_inputs[:, :, -kmin:], axis=2)),
                      lambda: outputs)
    outputs = tf.reshape(outputs, [batch_size, n_channels])

    # Kept indices of retained values
    indices = tf.cond(tf.greater(kmin, 0),
                      lambda: tf.concat([indices[:, :, :kmax], indices[:, :, -kmin:]], 2),
                      lambda: indices[:, :, :kmax])

    def grad(output_grad):
        # Prepare the indices for scatter operation
        i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(n_channels), indexing="ij")
        i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, kmin+kmax])
        i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, kmin+kmax])
        idx = tf.stack([i1, i2, indices], axis=-1)

        # Prepare the shape
        to_shape = [batch_size, n_channels, n]

        # Prepare the update tensor
        output_grad = tf.tile(output_grad[:, :, tf.newaxis], [1, 1, kmin+kmax])

        # Transfert the gradient only to the parameters used for retained values
        input_grad = tf.scatter_nd(idx, output_grad, to_shape)

        # Reshape the tensor and reorder the axis
        input_grad = tf.reshape(input_grad, input_shape)
        input_grad = tf.transpose(input_grad, [0, 2, 3, 1])
        return input_grad, None, None

    return outputs, grad


class WeldonPooling2dLayer(Layer):
    """
    Layer which apply Weldon Pooling on a tensor.
    """
    def __init__(self, kmax, kmin, **kwargs):
        """Initialization."""
        super(WeldonPooling2dLayer, self).__init__(**kwargs)
        self.kmax = kmax
        self.kmin = kmin

    def build(self, input_shape):
        self.trainable_weights = []
        super(WeldonPooling2dLayer, self).build(input_shape)

    def call(self, inputs):
        """Apply the MinMax Pooling."""
        return weldon_pooling_2d(inputs, self.kmin, self.kmax)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])
