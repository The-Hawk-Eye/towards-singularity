import numpy as np


from .layers import *
from .layers_fast import *


"""
A convolutional network with the following architecture:

conv - relu - 2x2 max pool -
conv - relu - 2x2 max pool -
affine - relu - affine - softmax

The network operates on minibatches of data that have shape (N, C, H, W)
consisting of N images, each with height H and width W and with C input
channels.
"""
class ConvNetwork(object):
    def __init__(self, input_dim=(3, 32, 32), channels_1=16, channels_2=32, filter_size=3,
                 hidden_dim=64, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Inputs:
        - input_dim: Tuple (C, H, W) giving the size of input data.
        - channels_1: Number of filters to use in the first convolutional layer.
        - channels_2: Number of filters to use in the second convolutional layer.
        - filter_size: Width/height of filters to use in the convolutional layers.
        - hidden_dim: Number of units to use in the fully-connected hidden layer.
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization of weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialization of weights and biases for the filters of the convolutional layers.
        self.params["KW1"] = np.random.randn(channels_1, input_dim[0], filter_size, filter_size) * weight_scale
        self.params["Kb1"] = np.zeros(channels_1)
        self.params["KW2"] = np.random.randn(channels_2, channels_1, filter_size, filter_size) * weight_scale
        self.params["Kb2"] = np.zeros(channels_2)

        # Initialization of weights and biases for the fully-connected layers.
        conv_layer_size = channels_2 * input_dim[1] // 4 * input_dim[2] // 4
        self.params["W1"] = np.random.randn(conv_layer_size, hidden_dim) * weight_scale
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params["b2"] = np.zeros(num_classes)

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def _forward(self, X):
        """
        Input:
        - X: A numpy array of input data of shape (N, d_1, ..., d_k)

        Returns:
        - scores: A numpy array of shape (N, C) giving classification scores, where
          scores[i, j] is the classification score for X[i] and class j.
        - caches: A list of tuples. Each tuple holds the cached variables needed
          for the respective backward pass.
        """
        conv_param = {'stride': 1, 'pad': 1}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        caches = []

        # Layer 1 - CONV - ReLU - MaxPool
        out, cache = conv_forward_fast(X, self.params["KW1"], self.params["Kb1"], conv_param)
        caches.append(cache)
        out, cache = relu_forward(out)
        caches.append(cache)
        out, cache = max_pool_forward_fast(out, pool_param)
        caches.append(cache)

        # Layer 2 - CONV - ReLU - MaxPool
        out, cache = conv_forward_fast(out, self.params["KW2"], self.params["Kb2"], conv_param)
        caches.append(cache)
        out, cache = relu_forward(out)
        caches.append(cache)
        out, cache = max_pool_forward_fast(out, pool_param)
        caches.append(cache)

        # Layer 3 - FC - ReLU
        out, cache = affine_forward(out, self.params["W1"], self.params["b1"])
        caches.append(cache)
        out, cache = relu_forward(out)
        caches.append(cache)

        # Layer 4 - FC
        scores, cache = affine_forward(out, self.params["W2"], self.params["b2"])
        caches.append(cache)    

        return scores, caches


    def _backward(self, y, scores, caches):
        """
        Inputs:
        - y: A numpy array of labels, of shape (N,). y[i] gives the label for X[i].
        - scores: A numpy array of shape (N, C) giving classification scores.
        - caches: A list of tuples. Each tuple holds cached variables.

        Returns:
        - loss: A scalar value giving the loss.
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        loss, dscores = cross_entropy_loss(scores, y)

        # Layer 4
        dout, dW2, db2 = affine_backward(dscores, caches.pop())
        loss += 0.5 * self.reg * np.sum(self.params["W2"])
        dW2 += self.reg * self.params["W2"]

        # Layer 3
        dout = relu_backward(dout, caches.pop())
        dout, dW1, db1 = affine_backward(dout, caches.pop())
        loss += 0.5 * self.reg * np.sum(self.params["W1"])
        dW1 += self.reg * self.params["W1"] 

        # Layer 2
        dout = max_pool_backward_fast(dout, caches.pop())
        dout = relu_backward(dout, caches.pop())
        dout, dKW2, dKb2 = conv_backward_fast(dout, caches.pop())
        loss += 0.5 * self.reg * np.sum(self.params["KW2"])
        dKW2 += self.reg * self.params["KW2"]

        # Layer 1
        dout = max_pool_backward_fast(dout, caches.pop())
        dout = relu_backward(dout, caches.pop())
        _, dKW1, dKb1 = conv_backward_fast(dout, caches.pop())
        loss += 0.5 * self.reg * np.sum(self.params["KW1"])
        dKW1 += self.reg * self.params["KW1"]

        grads = {"KW1":dKW1, "Kb1":dKb1, "KW2":dKW2, "Kb2":dKb2,
                 "W1":dW1, "b1":db1, "W2":dW2, "b2":db2}

        return loss, grads


    def loss(self, X, y=None):
        """
        Inputs:
        - X: A numpy array of input data of shape (N, d_1, ..., d_k)
        - y: A numpy array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - y_pred: A numpy array of shape (N, ) giving prediction labels, where
          y_pred[i] is the predicted label for X[i].

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: A scalar value giving the loss.
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        # Cast the input data to the same datatype as the parameters of the model.
        X = X.astype(self.dtype)

        # forward pass
        scores, caches = self._forward(X)

        # If test mode, return without computing the backward pass.
        if y is None:
            y_pred = np.argmax(scores, axis=1)
            return y_pred


        # backward pass
        loss, grads = self._backward(y, scores, caches)

        return loss, grads

#