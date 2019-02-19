"""Tests for custom pooling layer."""

# Standard imports
import unittest

# External imports
import numpy as np
import keras
import tensorflow as tf

# Internal imports
from weldon.weldon_pooling import weldon_pooling_2d, WeldonPooling2dLayer

class TestWeldonPooling2d(unittest.TestCase):

    def _expected_forward(self, inputs, kmin, kmax):
        """
        Compute the expected forward propagation output for WeldonPooling2d with Numpy.

        Parameters:
        -----------
        inputs : array of shape (batch_size, n, m, c)
            inputs of the layer during the forward propagation.
        kmin : int
            number of elements of minimal value kept by the layer.
        kmax : int
            number of elements of maximal value kept by the layer.
        """
        batch_size, _, _, n_channels = inputs.shape
        inputs = np.moveaxis(inputs, 3, 1)
        inputs = inputs.reshape((batch_size, n_channels, -1))

        sorted_inputs = np.sort(inputs, axis=-1)[:, :, ::-1] # descending order
        outputs = sorted_inputs[:, :, :kmax].mean(axis=2)
        if kmin > 0:
            outputs += sorted_inputs[:, :, -kmin:].mean(axis=2)

        return outputs.reshape((batch_size, n_channels))

    def _expected_grad(self, inputs, kmin, kmax):
        """
        Compute the expected backward propagation output for WeldonPooling2d with Numpy.

        Parameters:
        -----------
        inputs : array of shape (batch_size, n, m, c)
            inputs of the layer during the forward propagation.
        kmin : int
            number of elements of minimal value kept by the layer.
        kmax : int
            number of elements of maximal value kept by the layer.
        """
        batch_size, n, m, n_channels = inputs.shape
        inputs = np.moveaxis(inputs, 3, 1)
        inputs = inputs.reshape((batch_size, n_channels, -1))

        indices = np.argsort(inputs, axis=-1)[:, :, ::-1] # descending order
        indices = np.concatenate((indices[:, :, :kmax], indices[:, :, -kmin:]), axis=2)

        expected_grad = np.zeros_like(inputs)
        for i in range(batch_size):
            for j in range(n_channels):
                expected_grad[i, j, indices[i, j]] = 1.

        expected_grad = expected_grad.reshape((batch_size, n_channels, n, m))
        expected_grad = np.moveaxis(expected_grad, 1, 3)

        return expected_grad

    def test_forward_operation(self):
        inputs = np.random.random((10, 4, 4, 2))
        input_layer = tf.Variable(inputs, dtype='float32')

        output = weldon_pooling_2d(input_layer, 2, 2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual_output = sess.run(output)

        expected_output = self._expected_forward(inputs, 2, 2)

        self.assertTrue(np.all(np.isclose(expected_output, actual_output)))

    def test_backward_operation(self):
        inputs = np.random.random((10, 4, 4, 2))
        input_layer = tf.Variable(inputs, dtype='float32')
        output = weldon_pooling_2d(input_layer, 2, 2)
        grad = tf.gradients(output, input_layer)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual_grad = sess.run(grad)[0]

        expected_grad = self._expected_grad(inputs, 2, 2)

        self.assertTrue(np.all(np.isclose(expected_grad, actual_grad)))

    def test_integration_into_keras_layer(self):
        l_input = keras.layers.Input(shape=(4, 4, 2))
        model = keras.models.Model(inputs=l_input, outputs=WeldonPooling2dLayer(2, 2)(l_input))

        model.compile(optimizer="sgd", loss="mean_squared_error")

        X = np.random.random((10, 4, 4, 2))
        y = np.random.random((10, 2))

        model.fit(X, y)

if __name__ == "__main__":
    unittest.main()
