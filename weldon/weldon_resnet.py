"""
Implementation of Weldon network with resnet as features extractor.
"""

# Standard imports

# External librairies imports
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2D
from keras.models import Model

# Internal imports
from weldon.weldon_pooling import WeldonPooling2dLayer

def weldon(inputs, model, num_classes, pooling):
    """
    Instanciate the Weldon model.

    Parameters
    ----------
    inputs: keras.Input (BATCH_SIZE x N x M x N_CHANNELS)

    model: keras.Model
        model used to extract features
    num_classes: int
        number of classes.
    pooling : keras.Layer
        Layer used to pool a global score.

    Return
    ------
    x : keras.Model (! not compiled !)
    """
    # Features extraction
    x = model(inputs)

    # Classifier
    x = Conv2D(filters=num_classes, kernel_size=(1, 1), strides=1, use_bias=True)(x)

    # Pooling
    x = pooling(x)

    return Model(inputs, x)

def resnet50_weldon(num_classes, weights="imagenet", kmax=1, kmin=0):
    model = ResNet50(include_top=False, weights=weights, input_shape=(224, 224, 3), pooling=None)
    pooling = WeldonPooling2dLayer(kmax=kmax, kmin=kmin)
    inputs = Input(shape=(224, 224, 3))
    return weldon(inputs, model, num_classes, pooling)
