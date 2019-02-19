"""Tests for weldon model."""

# Standard imports
import unittest

# External imports
import numpy as np
import keras
import tensorflow as tf

# Internal imports
from weldon.weldon_resnet import resnet50_weldon

class TestWeldonResnet(unittest.TestCase):

    def test_resnet50_weldon(self):
        X = np.random.random((1, 224, 224, 3))
        y = np.random.randint(2, size=(1, 1))

        model = resnet50_weldon(2, kmax=2, kmin=2)
        model.compile(optimizer="sgd", loss="mean_squared_error")

        model.fit(X, keras.utils.to_categorical(y, 2))
