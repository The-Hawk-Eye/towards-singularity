"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights.
For each update rule a dictionary containing hyperparameter values and
caches of moving averages is also passed as a parameter.
"""


import numpy as np


"""
The function "sgd" performs plain vanilla stochastic gradient descent.
"""
def sgd(w, dw, config=None):
    """
    Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as w giving the gradient with respect to w.
    - config: A dictionary containing hyperparameter values:
        = learning_rate: Scalar learning rate.

    Returns:
    - next_w: A numpy array giving the updated parameters.
    - config: The config dictionary to be passed to the next iteration of the update rule.
    """
    if config is None: config = {}

    learning_rate = config.setdefault("learning_rate", 1e-2)
    next_w = w - learning_rate * dw

    return next_w, config


"""
The function "sgd_momentum" performs stochastic gradient descent with momentum.
"""
def sgd_momentum(w, dw, config=None):
    """
    Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as w giving the gradient with respect to w.
    - config: A dictionary containing hyperparameter values:
        = learning_rate: Scalar learning rate.
        = momentum: Scalar between 0 and 1 giving the momentum value.
        = velocity: A numpy array of the same shape as w used to store a moving average of the gradients.

    Returns:
    - next_w: A numpy array giving the updated parameters.
    - config: The config dictionary to be passed to the next iteration of the update rule.
    """
    if config is None: config = {}

    learning_rate = config.setdefault("learning_rate", 1e-2)
    mu = config.setdefault("momentum", 0.9)
    v = config.setdefault("velocity", np.zeros_like(w))

    v = mu * v - learning_rate * dw
    next_w = w + v

    config["velocity"] = v

    return next_w, config


"""
The function "sgd_nesterov" performs stochastic gradient descent using the
Nesterov Accelerated Gradient update.
"""
def sgd_nesterov(w, dw, config=None):
    """
    Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as w giving the gradient with respect to w.
    - config: A dictionary containing hyperparameter values:
        = learning_rate: Scalar learning rate.
        = momentum: Scalar between 0 and 1 giving the momentum value.
        = velocity: A numpy array of the same shape as w used to store a moving average of the gradients.

    Returns:
    - next_w: A numpy array giving the updated parameters.
    - config: The config dictionary to be passed to the next iteration of the update rule.
    """
    if config is None: config = {}

    learning_rate = config.setdefault("learning_rate", 1e-2)
    mu = config.setdefault("momentum", 0.9)
    v = config.setdefault("velocity", np.zeros_like(w))

    v_prev = v
    v = mu * v - learning_rate * dw
    next_w = w - mu * v_prev + (1 + mu) * v

    config["velocity"] = v

    return next_w, config


"""
The function "rmsprop" uses the RMSProp update rule. This rule uses a moving average
of squared gradient values to set adaptive per-parameter learning rates.
"""
def rmsprop(w, dw, config=None):
    """
    Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as w giving the gradient with respect to w.
    - config: A dictionary containing hyperparameter values:
        = learning_rate: Scalar learning rate.
        = decay_rate: Scalar between 0 and 1 giving the decay_rate of the squared gradient cache.
        = epsilon: A small scalar used for smoothing to avoid dividing by zero.
        = cache: A moving average of the squared gradients.

    Returns:
    - next_w: A numpy array giving the updated parameters.
    - config: The config dictionary to be passed to the next iteration of the update rule.
    """
    if config is None: config = {}

    learning_rate = config.setdefault("learning_rate", 1e-2)
    decay_rate = config.setdefault("decay_rate", 0.99)
    epsilon = config.setdefault("epsilon", 1e-8)
    cache = config.setdefault("cache", np.zeros_like(w))

    cache = decay_rate * cache + (1 - decay_rate) * (dw ** 2)
    next_w = w - learning_rate * dw / (np.sqrt(cache) + epsilon)

    config["cache"] = cache

    return next_w, config



"""
The function "adam" uses the Adam update rule. This rule incorporates moving averages
of both the gradient and its square. The rule also uses a bias correction term.
"""
def adam(w, dw, config=None):
    """
    Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as w giving the gradient with respect to w.
    - config: A dictionary containing hyperparameter values:
        = learning_rate: Scalar learning rate.
        = beta1: Scalar between 0 and 1 giving the decay rate of the gradient cache.
        = beta2: Scalar between 0 and 1 giving the decay rate of the squared gradient cache.
        = epsilon: A small scalar used for smoothing to avoid dividing by zero.
        = m: A moving average of the gradients.
        = v: A moving average of the squared gradients.
        = t: Iteration number.

    Returns:
    - next_w: A numpy array giving the updated parameters.
    - config: The config dictionary to be passed to the next iteration of the update rule.
    """
    if config is None: config = {}

    learning_rate = config.setdefault("learning_rate", 1e-3)
    beta1 = config.setdefault("beta1", 0.9)
    beta2 = config.setdefault("beta2", 0.999)
    epsilon = config.setdefault("epsilon", 1e-8)
    m = config.setdefault('m', np.zeros_like(w))
    v = config.setdefault('v', np.zeros_like(w))
    t = config.setdefault('t', 1)

    m = beta1 * m + (1 - beta1) * dw
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    vt = v / (1 - beta2 ** t)
    next_w = w - learning_rate * mt / (np.sqrt(vt) + epsilon)

    config['t'] += 1
    config['m'] = m
    config['v'] = v

    return next_w, config

#