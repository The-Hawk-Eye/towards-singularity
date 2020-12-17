"""
*******
The "forward" functions compute the forward pass for the given transformation.
Every forward function takes as input x:
- x: the input data
- parameters needed for computing the transformation
Every forward function returns a tuple (out, cache), where:
- out: is the output value of the layer.
- cache: is a tuple of cached variables, needed for computing the backward pass.

The "backward" functions compute the backward pass for the given transformation.
Every backward function takes as input dout and cahce:
- dout: is the upstream derivative.
- cache: is a tuple of cached variables, needed for computing the backward pass.
Every backward function returns:
- dx: the gradient with respect to the input x to the forward pass.
- dparams: the gradients with respect to the parameters of the forward layer.
*******
"""


import numpy as np


"""
An affine layer is a fully-connected layer mapping (N, D)-dimensional input
into (N, M)-dimensional output through an affine transformation.
An affine transformation is the composition of two functions:
- a linear map, represented by matrix multiplication
- translation, represented by vector addition
The function "affine_forward" computes the forward pass for an affine layer
and the function "affine_backward" computes the backward pass.
"""
def affine_forward(x, w, b=0):
    """
    Inputs:
    - x: A numpy array of shape (N, d_1, d_2, ..., d_k) containing mini-batch of N examples.
    - w: A numpy array of shape (D, M) containing the weights of the layer.
    - b: A numpy array of shape (M, ) containing the biases.

    Returns:
    - out: A numpy array of shape (N, M) containing the output values of the neurons.
    - cache: A tuple of variables needed for computing the backward pass.
    """
    _x = x.reshape(x.shape[0], -1)
    out = np.dot(_x, w) + b
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivative of size (N, M).
    - cache: A tuple (x, w, b) from affine_forward.

    Returns:
    - dx: Gradient with respect to x, of shape (N, d_1, d_2, ..., d_k).
    - dw: Gradient with respect to w, of shape (D, M).
    - db: Gradient with respect to b, of shape (M, ).
    """
    x, w, b = cache
    _x = x.reshape(x.shape[0], -1)

    db = np.sum(dout, axis=0)
    dw = np.dot(_x.T, dout)
    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)

    return dx, dw, db


################################
##### NON-LINEARITY LAYERS #####
################################
"""
Non-linear layers are inserted after every fully-connected layer applying
non-linearity to the network.
The function "*_forward" computes the forward pass for a non-linearity layer
and the function "*_backward" computes the backward pass.
The sigmoid layer uses the sigmoid non-linearity function (Sig(x) = 1 / (1 + exp(-x))).
The tanh layer uses the tanh non-linearity function.
The ReLU layer uses the rectified linear unit non-linearity truncating every
negative value to 0. (ReLU(x) = max(0, x)).
"""
def relu_forward(x):
    """
    Input:
    - x: Inputs, of any shape.

    Returns:
    - out: Output, of the same shape as x.
    - cache: x.
    """
    out = np.maximum(0, x)
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of any shape.
    - cache: Input x, of same shape as dout.

    Returns:
    - dx: Gradient with respect to the input x.
    """
    x = cache
    dx = dout * (x > 0).astype(int)

    return dx


def sigmoid_forward(x):
    """
    Input:
    - x: Inputs, of any shape.

    Returns:
    - out: Output, of the same shape as x.
    - cache: out.
    """
    out = 1 / (1 + np.exp(-x))
    cache = out

    return out, cache


def sigmoid_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of any shape.
    - cache: Input x, of same shape as dout.

    Returns:
    - dx: Gradient with respect to the input x.
    """
    out = cache
    dx = dout * out * (1 - out)

    return dx


def tanh_forward(x):
    """
    Input:
    - x: Inputs, of any shape.

    Returns:
    - out: Output, of the same shape as x.
    - cache: out.
    """
    out = (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))
    cache = out

    return out, cache


def tanh_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of any shape.
    - cache: Input x, of same shape as dout.

    Returns:
    - dx: Gradient with respect to the input x.
    """
    out = cache
    dx = dout * (1 - out ** 2)

    return dx


################################
######## DROPOUT LAYERS ########
################################
"""
Dropout layers are inserted before affine layers.
The function "dropout_forward" computes the forward pass for a droput layer.
During training every neuron is kept active with some probability 'p', otherwise
it is set to 0. The neurons that are kept active are scaled by a factor "1/p" in
order to preserve their expected output. This technique is known as "inverted dropout".
During test time all the neurons are kept active.
The function "dropout_backward" computes the backward pass for a dropout layer.
"""
def dropout_forward(x, dropout_param):
    """
    Inputs:
    - x: Input data, of any shape.
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking.

    Returns:
    - out: A numpy array of the same shape as x.
    - cache: A tuple of variables needed for computing the backward pass.
    """
    p = dropout_param['p'], 
    mode = dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    if mode == "train":
        mask = np.random.uniform(0, 1, x.shape) < p
        out = x * mask / p

    elif mode == "test":
        # mask = np.ones(x.shape) * p
        mask = None
        out = x

    else:
        raise ValueError("Invalid forward dropout mode '%s'" % mode)

    out = out.astype(x.dtype, copy=False)
    cache = (dropout_param, mask)

    return out, cache


def dropout_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of any shape.
    - cache: A tuple (dropout_param, mask) from dropout_forward.

    Returns:
    - dx: Gradient with respect to the input x.
    """
    dropout_param, mask = cache
    p = dropout_param['p'], 
    mode = dropout_param["mode"]

    if mode == "train":
        dx = dout * mask / p

    elif mode == "test":
        dx = dout

    else:
        raise ValueError("Invalid forward dropout mode '%s'" % mode)

    return dx


################################
##### NORMALIZATION LAYERS #####
################################
"""
A Normalization layer is inserted immediately after fully-connected layers and
before non-linearities.
The function "batchnorm_forward" during training computes the sample mean and
sample variance from minibatch statistics and normalizes the incoming data. It
also keeps a running mean and a runnig variance for each feature.
At test time the data is normalized using the running averages.
The functions "layernorm_forward" and "groupnorm_forward" compute the mean and the
variance over the features per data-point and normalize the incoming data in the
same way during training and testing. Thus, we dont keep running averages for
these functions.
The normalization layers also include learnable shift and scale parameters
for each feature.
The functions "*norm_backward" compute the backward pass through the Normalization
layers and compute the gradient with respect to the shift and scale parameters as
well as the gradient with respect to the input data.
"""
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Inputs:
    - x: A numpy array of shape (N, D).
    - gamma: Scale parameter of shape (D,).
    - beta: Shift paremeter of shape (D,).
    - bn_param: Dictionary with the following keys:
      - mode: "train" or "test"; required.
      - eps: Constant for numeric stability.
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features.
      - running_var Array of shape (D,) giving running variance of features.

    Returns:
    - out: A numpy array of shape (N, D) containing the output values of the neurons.
    - cache: A tuple of variables needed for computing the backward pass.
    """
    N, D = x.shape
    mode = bn_param["mode"]
    eps = bn_param.setdefault("eps", 1e-5)
    momentum = bn_param.setdefault("momentum", 0.9)
    running_mean = bn_param.setdefault("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.setdefault("running_var", np.zeros(D, dtype=x.dtype))

    if mode == "train":
        sample_mean = np.sum(x, axis=0) / N
        sample_var = np.sum((x - sample_mean) ** 2, axis=0) / N
        sample_std = np.sqrt(sample_var + eps)
        x_hat = (x - sample_mean) / sample_std
        out = gamma * x_hat + beta
        cache = (x, x_hat, gamma, sample_std)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

    elif mode == "test":
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        cache = None

    else:
        raise ValueError("Invalid forward batchnorm mode '%s'" % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of shape (N, D).
    - cache: A tuple (x, x_hat, gamma, sample_std) from batchnorm_forward.

    Returns:
    - dx: Gradient with respect to the input x, of shape (N, D).
    - dgamma: Gradient with respect to the scale parameter gamma, of shape (D,).
    - dbeta: Gradient with respect to the shift parameter beta, of shape (D,).
    """
    N = dout.shape[0]
    x, x_hat, gamma, sample_std = cache

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dx = (dout - (np.sum(dout, axis=0) + np.sum(dout * x_hat, axis=0) * x_hat) / N) * (gamma / sample_std)

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Inputs:
    - x: A numpy array of shape (N, C, H , W).
    - gamma: Scale parameter of shape (C,).
    - beta: Shift paremeter of shape (C,).
    - bn_param: Dictionary with the following keys.
      - mode: "train" or "test"; required.
      - eps: Constant for numeric stability.
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features.
      - running_var Array of shape (D,) giving running variance of features.

    Returns:
    - out: A numpy array of shape (N, C, H, W) containing the output values of the neurons.
    - cache: A tuple of variables needed for computing the backward pass.
    """
    N, C, H, W = x.shape
    _x = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(_x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)


    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W).
    - cache: A tuple of variables from spatial_batchnorm_forward.

    Returns:
    - dx: Gradient with respect to the input x, of shape (N, C, H, W).
    - dgamma: Gradient with respect to the scale parameter gamma, of shape (C,).
    - dbeta: Gradient with respect to the shift parameter beta, of shape (C,).
    """
    N, C, H, W = dout.shape
    _dout = (dout.transpose(0, 2, 3, 1)).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(_dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Inputs:
    - x: A numpy array of shape (N, D).
    - gamma: Scale parameter of shape (D,).
    - beta: Shift paremeter of shape (D,).
    - ln_param: Dictionary with the following keys:
      - eps: Constant for numeric stability.

    Returns:
    - out: A numpy array of shape (N, D) containing the output values of the neurons.
    - cache: A tuple of variables needed for computing the backward pass.
    """
    N, D = x.shape
    eps = ln_param.setdefault("eps", 1e-5)

    feature_mean = np.sum(x, axis=1, keepdims=True) / D
    feature_var = np.sum((x - feature_mean) ** 2, axis=1, keepdims=True) / D
    feature_std = np.sqrt(feature_var + eps)

    x_hat = (x - feature_mean) / feature_std
    out = gamma * x_hat + beta

    cache = (x, gamma, feature_mean, feature_var, feature_std, x_hat)

    return out, cache


def layernorm_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of shape (N, D).
    - cache: A tuple of intermediate variables from layernorm_forward.

    Returns:
    - dx: Gradient with respect to the input x, of shape (N, D).
    - dgamma: Gradient with respect to the scale parameter gamma, of shape (D,).
    - dbeta: Gradient with respect to the shift parameter beta, of shape (D,).
    """
    N, D = dout.shape
    x, gamma, feature_mean, feature_var, feature_std, x_hat = cache

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    dx_hat = dout * gamma
    dfeature_std = np.sum(dx_hat * (x - feature_mean), axis=1, keepdims=True) * (1 / (feature_std ** 2))
    dfeature_var = dfeature_std * (-1 / (2 * feature_std))
    dfeature_mean = np.sum(dx_hat, axis=1, keepdims=True) * (-1 / feature_std) - dfeature_var * (2 / D) * np.sum(x - feature_mean, axis=1, keepdims=True)
    dx = dx_hat * (1 / feature_std) + dfeature_mean * (1 / D) + dfeature_var * (2 * (x - feature_mean) / D)
    # dx = (dout - (np.sum(dout, axis=1, keepdims=True) + np.sum(dout * x_hat, axis=1, keepdims=True) * x_hat) / D) * (gamma / feature_std)

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Inputs:
    - x: A numpy array of shape (N, C, H, W).
    - gamma: Scale parameter of shape (C,).
    - beta: Shift paremeter of shape (C,).
    - G: Integer mumber of groups to split into, should be a divisor of C.
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability.

    Returns:
    - out: A numpy array of shape (N, C, H, W) containing the output values of the neurons.
    - cache: A tuple of variables needed for computing the backward pass.
    """
    N, C, H, W = x.shape
    eps = gn_param.setdefault("eps", 1e-5)

    _x = x.reshape(N, G, C // G, H, W)
    _x = x.reshape(N * G, C // G * H * W)

    feature_mean = np.sum(_x, axis=1, keepdims=True) / (C // G * H * W)
    feature_var = np.sum((_x - feature_mean) ** 2, axis=1, keepdims=True) / (C // G * H * W)
    feature_std = np.sqrt(feature_var + eps)

    x_hat = (_x - feature_mean) / feature_std
    x_hat = x_hat.reshape(N, G, C // G, H, W)
    x_hat = x_hat.reshape(N, C, H, W)
    out = gamma * x_hat + beta

    cache = (_x, gamma, feature_mean, feature_var, feature_std, x_hat, G)

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W).
    - cache: A tuple of intermediate variables from spatial_groupnorm_forward.

    Returns:
    - dx: Gradient with respect to the input x, of shape (N, C, H, W).
    - dgamma: Gradient with respect to the scale parameter gamma, of shape (C,).
    - dbeta: Gradient with respect to the shift parameter beta, of shape (C,).
    """
    N, C, H, W = dout.shape
    _x, gamma, feature_mean, feature_var, feature_std, x_hat, G = cache

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)

    dx_hat = dout * gamma
    dx_hat = dx_hat.reshape(N, G, C // G, H, W)
    dx_hat = dx_hat.reshape(N * G, C // G * H * W)
    dfeature_std = np.sum(dx_hat * (_x - feature_mean), axis=1, keepdims=True) * (1 / (feature_std ** 2))
    dfeature_var = dfeature_std * (-1 / (2 * feature_std))
    dfeature_mean = np.sum(dx_hat, axis=1, keepdims=True) * (-1 / feature_std) - dfeature_var * (2 / (C // G * H * W)) * np.sum(_x - feature_mean, axis=1, keepdims=True)
    dx = dx_hat * (1 / feature_std) + dfeature_mean * (1 / (C // G * H * W)) + dfeature_var * (2 * (_x - feature_mean) / (C // G * H * W))
    dx = dx.reshape(N, G, C // G, H, W)
    dx = dx.reshape(N, C, H, W)

    return dx, dgamma, dbeta


################################
##### CONVOLUTIONAL LAYERS #####
################################
"""
A convolutional layer consits of a set of filters shared between neurons.
During the forward pass each filter is slided across the spatial dimensions
of the input to produce an activation map using dot products. Each filter
produces a separate activation map, and stacking these maps together we
produce the output.
The functions "conv_forward" compute the forward pass for a convolutional layer.
The backward pass of a convolutional layer is also a convolution operation.
The functions "conv_backward" compute the backward pass through the Convolutional
layers and compute the gradient with respect to the filter parameters as well as
the gradient with respect to the input data.
"""
def conv_2d_forward_naive(x, w, b, conv_param):
    """
    Input:
    - x: A numpy array of shape (N, C, H, W).
    - w: A numpy array of shape (F, C, HH, WW) containing filter weights.
    - b: A numpy array of shape (F,) containing biases.
    - conv_param: A dictionary with the following keys:
      - stride: The number of pixels between adjacent receptive fields in the spatial directions.
      - pad: The number of pixels that will be used to zero-pad the input.
      - dilation: The number of pixels between each cell of the filter.

    Returns:
    - out: A numpy array of shape (N, F, H', W') containing the output values of the neurons.
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: A tuple of variables needed for computing the backward pass.
    """
    stride = conv_param.setdefault("stride", 1)
    pad = conv_param.setdefault("pad", 0)
    dilation = conv_param.setdefault("dilation", 0)
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    x_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    x_pad[:, :, pad : H + pad, pad : W + pad] = x
    x_pad = x_pad[:, np.newaxis, :]

    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride

    out = np.ndarray((N, F, H_prime, W_prime))
    for _r in range(H_prime):
        for _c in range(W_prime):
            _y = x_pad[:, :, :, _r * stride : HH + _r * stride, _c * stride : WW + _c * stride] * w
            out[:, :, _r, _c] = np.sum(_y, axis=(2, 3, 4)) + b

    cache = (x, w, b, conv_param)
    return out, cache


def conv_2d_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W).
    - cache: A tuple (x, w, b, conv_param) from conv_2d_forward

    Returns:
    - dx: Gradient with respect to the input x, of shape (N, C, H', W')
    - dw: Gradient with respect to the filter weights w, of shape (F, C, HH, WW)
    - db: Gradient with respect to the biases b, of shape (F,)
    """
    x, w, b, conv_param = cache
    _, _, filter_size, _ = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    # dilation = conv_param["dilation"]

    db = np.sum(dout, axis=(0,2,3))
    # works only with stride = 1 !!!
    # stride = 2 probably implies dilation = 2 !
    conv_param_dw = {"stride" : stride, "pad" : pad}
    dw, _ = conv_2d_forward_naive(x.transpose(1,0,2,3), dout.transpose(1,0,2,3), 0, conv_param_dw)
    dw = dw.transpose(1,0,2,3)

    conv_param_dx = {"stride" : stride, "pad" : filter_size - pad - 1}
    dx, _ = conv_2d_forward_naive(dout.transpose(0,1,2,3), np.flip(w.transpose(1,0,2,3), (2,3)), 0, conv_param_dx)

    return dx, dw, db


def max_pool_2d_forward(x, pool_param):
    """
    Inputs:
    - x: A numpy array of shape (N, C, H, W).
    - pool_param: dictionary with the following keys:
      - pool_height: The height of each pooling region.
      - pool_width: The width of each pooling region.
      - stride: The distance between adjacent pooling regions.

    Returns:
    - out: A numpy array of shape (N, C, H', W') containing the output values of the neurons.
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: A tuple of variables needed for computing the backward pass.
    """
    pool_height = pool_param.setdefault("pool_height", 1)
    pool_width = pool_param.setdefault("pool_width", 1)
    stride = pool_param.setdefault("stride", 1)

    N, C, H, W = x.shape
    H_prime = 1 + (H - pool_height) // stride
    W_prime = 1 + (W - pool_width) // stride

    out = np.ndarray((N, C, H_prime, W_prime))
    for _r in range(H_prime):
        for _c in range(W_prime):
            out[:, :, _r, _c] = np.amax(x[:, :, _r*stride : _r*stride + pool_height, _c*stride : _c*stride + pool_width], axis=(2,3))

    cache = (x, pool_param)
    return out, cache


def max_pool_2d_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W).
    - cache: A tuple (x, pool_param) from max_pool_2d_forward.

    Returns:
    - dx: Gradient with respect to the input x, of shape (N, C, H, W).
    """
    x, pool_param = cache
    pool_height = pool_param.setdefault("pool_height", 1)
    pool_width = pool_param.setdefault("pool_width", 1)
    stride = pool_param.setdefault("stride", 1)

    N, C, H_prime, W_prime = dout.shape

    dx = np.zeros_like(x)
    for _n in range(N):
        for _ch in range(C):
            for _r in range(H_prime):
                for _c in range(W_prime):
                    arg = np.argmax(x[_n, _ch, _r*stride : _r*stride + pool_height, _c*stride : _c*stride + pool_width])
                    arg1 = arg // pool_width + _r * stride
                    arg2 = arg % pool_width + _c * stride
                    dx[_n, _ch, arg1, arg2] += dout[_n, _ch, _r, _c]

    return dx


################################
####### RECURRENT LAYERS #######
################################
"""
At each time-step, there are two inputs to the hidden layer:
 - the input at that time-step x_i
 - the output of the previous hidden layer h_(i-1)
The former is multiplied by a weight matrix Wx and the latter by a weight
matrix Wh. The sum of the two is run through a cell non-linearity function.
The function "*_step_forward" computes the forward pass for a recurrent layer
and the function "*_step_backward" computes the backward pass.
"""
def recurrent_step_forward(x, prev_h, Wx, Wh, b):
    """
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H).
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H).
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H).
    - b: Biases of shape (H,).

    Returns:
    - next_h: Next hidden state, of shape (N, H).
    - cache: Tuple of values needed for the backward pass.
    """
    next_h = x.dot(Wx) + prev_h.dot(Wh) + b
    next_h = np.tanh(next_h)

    cache = (x, Wx, Wh, prev_h, next_h)

    return next_h, cache


def recurrent_step_backward(dnext_h, cache):
    """
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H).
    - cache: Cache object from the forward pass.

    Returns:
    - dx: Gradients of input data, of shape (N, D).
    - dprev_h: Gradients of previous hidden state, of shape (N, H).
    - dWx: Gradients of input-to-hidden weights, of shape (D, H).
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H).
    - db: Gradients of bias vector, of shape (H,).
    """
    x, Wx, Wh, prev_h, next_h = cache

    dtanh = dnext_h * (1 - next_h ** 2)
    dx = dtanh.dot(Wx.T)
    dprev_h = dtanh.dot(Wh.T)
    dWx = x.T.dot(dtanh)
    dWh = prev_h.T.dot(dtanh)
    db = np.sum(dtanh, axis=0)

    return dx, dprev_h, dWx, dWh, db


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Inputs:
    - x: Input data, of shape (N, D).
    - prev_h: Previous hidden state, of shape (N, H).
    - prev_c: Previous cell state, of shape (N, H).
    - Wx: Input-to-hidden weights, of shape (D, 4H).
    - Wh: Hidden-to-hidden weights, of shape (H, 4H).
    - b: Biases, of shape (4H,).

    Returns:
    - next_h: Next hidden state, of shape (N, H).
    - next_c: Next cell state, of shape (N, H).
    - cache: Tuple of values needed for backward pass.
    """
    N, H = prev_h.shape
    h = x.dot(Wx) + prev_h.dot(Wh) + b

    # sigmoid = lambda x: 1 / (1 + np.exp(-x))

    i = sigmoid(h[:, : H])
    f = sigmoid(h[:, H : 2*H])
    o = sigmoid(h[:, 2*H : 3*H])
    g = np.tanh(h[:, 3*H : ])

    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    cache = (x, prev_h, prev_c, Wx, Wh, h, i, f, o, g, next_c)

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    N, H = dnext_h.shape
    x, prev_h, prev_c, Wx, Wh, h, i, f, o, g, next_c = cache

    dnext_c += dnext_h * o * (1 - np.tanh(next_c) ** 2)
    dprev_c = dnext_c * f

    di = dnext_c * g
    df = dnext_c * prev_c
    do = dnext_h * np.tanh(next_c)
    dg = dnext_c * i

    # sigmoid = lambda x: 1 / (1 + np.exp(-x))

    dh = np.hstack([di, df, do, dg])
    dh[:, : 3*H] = dh[:, : 3*H] * sigmoid(h[:, : 3*H]) * (1 - sigmoid(h[:, : 3*H]))
    dh[:, 3*H : ] = dh[:, 3*H :] * (1 - np.tanh(h[:, 3*H :]) ** 2)

    dx = dh.dot(Wx.T)
    dprev_h = dh.dot(Wh.T)
    dWx = x.T.dot(dh)
    dWh = prev_h.T.dot(dh)
    db = np.sum(dh, axis=0)

    return dx, dprev_h, dprev_c, dWx, dWh, db


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def recurrent_forward(x, h0, Wx, Wh, b):
    """
    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H, )

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    N, T, D = x.shape
    _, H = h0.shape
    h = np.ndarray((N, T, H))
    caches = []

    # Unroll the recurrent network step-by-step.
    prev_h = np.zeros((N, H)) if h0 is None else h0
    for t in range(T):
        # make one step
        next_h, next_cache = recurrent_step_forward(x[:, t, :], prev_h, Wx, Wh, b)

        # store the results
        h[:, t, :] = next_h
        caches.append(next_cache)

        # update the state
        prev_h = next_h

    caches.append(D)
    return h, caches


def recurrent_backward(dh, caches):
    """
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 

    Returns:
    - dx: Gradient with respect to the inputs, of shape (N, T, D)
    - dh0: Gradient with respect to the initial hidden state, of shape (N, H)
    - dWx: Gradient with respect to the  input-to-hidden weights, of shape (D, H)
    - dWh: Gradient with respect to the  hidden-to-hidden weights, of shape (H, H)
    - db: Gradient with respect to the biases, of shape (H, )
    """
    N, T, H = dh.shape
    D = caches.pop()

    dx = np.ndarray((N, T, D))
    dh0, dWx, dWh, db = 0, 0, 0, 0
    dprev_h = 0
    for t in range(T - 1, -1, -1):
        dx_step, dprev_h, dWx_step, dWh_step, db_step = recurrent_step_backward(dh[:, t, :] + dprev_h,
                                                                                caches.pop())
        dx[:, t, :] = dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step

    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H, )

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    N, T, D = x.shape
    _, H = h0.shape
    h = np.ndarray((N, T, H))
    caches = []

    # Unroll the recurrent network step-by-step.
    prev_h = np.zeros((N, H)) if h0 is None else h0
    prev_c = np.zeros((N, H))
    for t in range(T):
        # make one step
        next_h, next_c, next_cache = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)

        # store the results
        h[:, t, :] = next_h
        caches.append(next_cache)

        # update the state
        prev_h = next_h
        prev_c = next_c

    caches.append(D)
    return h, caches


def lstm_backward(dh, caches):
    """
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 

    Returns:
    - dx: Gradient with respect to the inputs, of shape (N, T, D)
    - dh0: Gradient with respect to the initial hidden state, of shape (N, H)
    - dWx: Gradient with respect to the  input-to-hidden weights, of shape (D, H)
    - dWh: Gradient with respect to the  hidden-to-hidden weights, of shape (H, H)
    - db: Gradient with respect to the biases, of shape (H, )
    """
    N, T, H = dh.shape
    D = caches.pop()

    dx = np.ndarray((N, T, D))
    dh0, dWx, dWh, db = 0, 0, 0, 0
    dprev_h = 0
    dprev_c = 0
    for t in range(T - 1, -1, -1):
        dx_step, dprev_h, dprev_c, dWx_step, dWh_step, db_step = lstm_step_backward(dh[:, t, :] + dprev_h,
                                                                                    dprev_c, caches.pop())
        dx[:, t, :] = dx_step
        dWx += dWx_step
        dWh += dWh_step
        db += db_step

    dh0 = dprev_h

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b=0):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: A numpy array of shape (N, T, D).
    - w: A numpy array of shape (D, M) containing the weights.
    - b: A numpy array of shape (M, ) containing the biases.

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass.
    """
    N, T, D = x.shape
    M = b.shape[0]

    out, cache = affine_forward(x.reshape(N * T, D), w, b)
    out = out.reshape(N, T, M)

    return out, (cache, D)


def temporal_affine_backward(dout, cache):
    """
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass.

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, T, D)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M, )
    """
    N, T, M = dout.shape
    cache, D = cache

    dx, dw, db = affine_backward(dout.reshape(N * T, M), cache)
    dx = dx.reshape(N, T, D)

    return dx, dw, db


################################
########## ATTENTION  ##########
################################
"""
At each time-step, attention scores are computed between the decoder hidden
state and the encoder annotations. The scores are converted to an attention
distribution using a softmax function. The attention output is a weighted sum
of the encoder annotations where the weight of each annotation is given by the
attention distribution. The function "attention_forward" computes the attention
output in the forward pass and the function "attention_backward" computes the
gradients in the backward pass.
"""
def attention_forward(h_t, h_enc):
    """
    Inputs:
    - h_t: A numpy array of shape (N, H) giving decoder hidden state at the
      current time-step.
    - h_enc: A numpy array of shape (N, T, H) giving encoder annotations.

    Returns:
    - out: A numpy array of shape (N, H) giving attention output.
    - cache: Values needed for the backward pass.
    """
    N, T, H = h_enc.shape

    # Compute the attention scores using dot-product.
    scores = np.sum(h_enc * h_t[:, np.newaxis, :], axis=-1)
    scores -= np.max(scores, axis=1, keepdims=True)

    # Compute the attention distribution using softmax.
    distr = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    # Compute the attention output.
    out = np.sum(distr[:, :, np.newaxis] * h_enc, axis=1)

    cache = (h_t, h_enc, scores, distr)
    return out, cache


def attention_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivative of the attention output of shape (N, H)
    - cache: Values from the forward pass.

    Returns:
    - dh_t: Gradient with respect to decoder hidden state of shape (N, H).
    - dh_enc: Gradient with respect to the encoder annotations of shape (N, T, H).
    """
    h_t, h_enc, scores, distr = cache

    d_distr = np.sum(dout[:, np.newaxis, :] * h_enc, axis=-1)
    dh_enc = dout[:, np.newaxis, :] * distr[:, :, np.newaxis]
    dscores = distr * (d_distr - np.sum(d_distr * distr, axis=1, keepdims=True))
    dh_enc += dscores[:, :, np.newaxis] * h_t[:, np.newaxis, :]
    dh_t = np.sum(dscores[:, :, np.newaxis] * h_enc, axis=1)

    return dh_t, dh_enc


def temporal_attention_forward(h_dec, h_enc):
    """
    Inputs:
    - h_dec: A numpy array of shape (N, T_d, H) giving decoder hidden states.
    - h_enc: A numpy array of shape (N, T_e, H) giving encoder annotations.

    Returns:
    - out: A numpy array of shape (N, T_d, H) giving attention output for every
      decoder hidden state.
    - cache: Values needed for the backward pass.
    """
    N, T_d, H = h_dec.shape

    # Compute the attention scores using dot-product. Shape (N, T_d, T_e)
    scores = np.sum(h_enc[:, np.newaxis, :, :] * h_dec[:, :, np.newaxis, :], axis=-1)
    scores = scores - np.max(scores, axis=2, keepdims=True)

    # Compute the attention distribution using softmax. Shape (N, T_d, T_e)
    distr = np.exp(scores) / np.sum(np.exp(scores), axis=2, keepdims=True)

    # Compute the attention output. Shape (N, T_d, H)
    out = np.sum(distr[:, :, :, np.newaxis] * h_enc[:, np.newaxis, :, :], axis=2)

    cache = (h_dec, h_enc, scores, distr)
    return out, cache


def temporal_attention_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivative of the attention output at every time-step of shape (N, T_dec, H)
    - cache: Values from the forward pass.

    Returns:
    - dh_dec: Gradient with respect to the decoder hidden states of shape (N, T_dec, H).
    - dh_enc: Gradient with respect to the encoder annotations of shape (N, T_enc, H).
    """
    h_dec, h_enc, scores, distr = cache

    d_distr = np.sum(dout[:, :, np.newaxis, :] * h_enc[:, np.newaxis, :, :], axis=-1)
    dh_enc = np.sum(dout[:, :, np.newaxis, :] * distr[:, :, :, np.newaxis], axis=1)
    dscores = distr * (d_distr - np.sum(d_distr * distr, axis=2, keepdims=True))#[:, :, np.newaxis])
    dh_enc += np.sum(dscores[:, :, :, np.newaxis] * h_dec[:, :, np.newaxis, :], axis=1)
    dh_dec = np.sum(dscores[:, :, :, np.newaxis] * h_enc[:, np.newaxis, :, :], axis=2)

    return dh_dec, dh_enc


################################
######## WORD EMBEDDING ########
################################
"""
Given a vocabulary of V words, we assign each word to a vector
of dimension D.
The forward pass transforms a batch of words into a batch of
vectors.
The backward pass computes the gradient of the loss with
respect to the embedding matrix.
"""
def word_embedding_forward(x, W):
    """
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: A numpy array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass.
    """
    N, T = x.shape

    out = W[x.flatten()]
    out = out.reshape(N, T, -1)
    cache = x, W

    return out, cache


def word_embedding_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass.

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    x, W = cache
    N, T, D = dout.shape

    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)

    return dW


################################
######## LOSS FUNCTIONS ########
################################
"""
The loss functions compute the data loss measure for the network.
The input consists of scores and labels.
Scores gives the computed score for every class for every training input.
Labels gives the correct class for every training input. 
The loss functions return a scalar giving the data loss and a numpy
array giving the gradient of the loss with respect to the scores.
"""
def cross_entropy_loss(scores, labels):
    """
    Inputs:
    - scores: A numpy array of shape (N, C), where scores[i, j] is the score of
      the j-th class for the i-th input.
    - labels: A numpy array of shape (N, ), where labels[i] is the correct class for x[i].

    Returns:
    - loss: Scalar giving the loss.
    - dscores: Gradient of the loss with respect to scores.
    """
    N, C = scores.shape
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)     # shifting
    Y = np.log(np.sum(np.exp(shifted_scores), axis=1, keepdims=True))   # log of sum of exp
    Z = shifted_scores - Y              # computes log(sum(exp(s_ik))) - s_ij, for every i and j
    loss = - np.sum(Z[np.arange(N), labels]) / N    # sums only the relevant terms

    dscores = np.exp(Z)                 # effectively computes the first part of the derivative
    dscores[np.arange(N), labels] -= 1  # computes the second part of the derivative
    dscores /= N

    return loss, dscores


def hinge_loss(scores, labels):
    """
    Inputs:
    - scores: A numpy array of shape (N, C), where scores[i, j] is the score of
      the j-th class for the i-th input.
    - labels: A numpy array of shape (N, ), where labels[i] is the correct class for x[i].

    Returns:
    - loss: Scalar giving the loss.
    - dscores: Gradient of the loss with respect to scores.
    """
    N, C = scores.shape
    correct_class_scores = scores[np.arange(N), labels]
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), labels] = 0
    loss = np.sum(margins) / N

    dscores = (margins > 0).astype(float)
    dscores[np.arange(N), labels] = -np.sum(dscores, axis=1)
    dscores /= N

    return loss, dscores


def negative_sampling_loss(scores, labels, neg_labels):
    """
    Inputs:
    - scores: A numpy array of shape (N, V), where scores[i, j] is the scores of
      the j-th class for input X[i].
    - labels: An integer array of shape (N, T), where each of labels[i, j] is a correct
      class for X[i].
    - neg_labels: An integer array of shape (N, K), where each of neg_labels[i, j] gives
      an incorrect class for X[i].

    Returns:
    - loss: Scalar giving the loss.
    - dscores: Gradient of the loss with respect to scores.
    """
    N, V = scores.shape
    N, K = neg_labels.shape

    pos = scores[np.arange(N).reshape(N, 1), labels]
    neg = scores[np.arange(N).reshape(N, 1), neg_labels]
    loss = - np.sum(np.log(sigmoid(pos))) - np.sum(np.log(sigmoid(-neg)))

    deriv_pos = lambda x: - (1 - sigmoid(x))
    deriv_neg = lambda x: (1 - sigmoid(-x))

    dscores = np.zeros_like(scores, dtype=np.float64)
    np.add.at(dscores, (np.arange(N).reshape(N,1), labels), deriv_pos(pos))
    np.add.at(dscores, (np.arange(N).reshape(N,1), neg_labels), deriv_neg(neg))

    return loss, dscores


def temporal_cross_entropy_loss(x, y, mask=None, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)

    # If no mask is provided then the full sequence contributes to the loss.
    if mask is None:
        mask = np.full((N, T), True, dtype=np.bool)

    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx