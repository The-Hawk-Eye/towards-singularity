import numpy as np
import layers
from layers import *


"""
A fully-connected neural network with an arbitrary number of hidden layers.
Dropout and Batch Normalization are implemented as optional.
The network uses softmax loss function.
For a network with L layers the architecture will be:

{affine - [BatchNorm] - Non-Linearity - [dropout]} x (L - 1) - affine - softmax

The non-linearity updates in the hidden layers that can be used are defined
in "layers.py".
Learnable parameters are stored in the self.params dictionary that maps
parameter names to numpy arrays.
The weights and biases for the first layer are stored under the keys "W1"
and "b1", for the second layer - "W2" and "b2", and so on.
When using batch normalization, scale and shift parameters for the first
layer are stored under the keys "gamma1" and "beta1", for the second
layer - "gamma2" and "beta2", and so on.

Training is performed using the Solver class.
"""
class NeuralNetwork(object):
	def __init__(self, hidden_dims, input_dim, output_dim,
				weight_scale=1e-3, reg=0.0,
				dropout=1,						# dropout != 1 - flag to use dropout
				normalization=None,				# normalization="batchnorm" - flag to use batchnorm
				nonlinearity="relu",			# non-linearity activation function
				dtype=np.float32, seed=None):
		"""
		Inputs:
		- hidden_dims: A list of integers giving the size of each hidden layer.
		- input_dim: An integer giving the size of the input.
		- output_dim: An integer giving the number of classes to classify.
		- weight_scale: Scalar giving the standard deviation for random
		  initialization of the weights.
		- reg: Scalar giving L2 regularization strength.
		- dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
		  the network should not use dropout at all.
		- normalization: Valid values are "batchnorm" for batch normalization and "None"
		  for no normalization.
		- nonlinearity: A string giving the name of a non-linearity rule. Default is "relu".
		- dtype: A numpy datatype object; all computations will be performed using
		  this datatype. float32 is faster but less accurate,
		  float64 is used for numeric gradient checking.
		- seed: If not None, then pass this random seed to the dropout layers. This
		  will make the dropout layers deterministic so we can gradient check the
		  model.
		"""
		self.num_layers = 1 + len(hidden_dims)
		self.reg = reg
		self.use_dropout = dropout != 1
		self.normalization = normalization
		self.dtype = dtype
		self.params = {}

		# Make sure the non-linearity rule exists, then use the actual function.
		if not hasattr(layers, nonlinearity + "_forward"):
			raise ValueError("Invalid nonlinearity '%s'" % nonlinearity)
		self.nonlin_forward = getattr(layers, nonlinearity + "_forward")
		self.nonlin_backward = getattr(layers, nonlinearity + "_backward")

		# Initialize the weights.
		dims = [input_dim] + hidden_dims + [output_dim]	
		if type(weight_scale) == float:
			self.params.update([("W%d" % (i + 1), np.random.randn(dims[i], dims[i + 1]) * weight_scale) for i in range(self.num_layers)])
		elif weight_scale == "Xavier":
			self.params.update([("W%d" % (i + 1), np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2 / (dims[i] + dims[i + 1]))) for i in range(self.num_layers)])
		elif weight_scale == "Kaiming":
			self.params.update([("W%d" % (i + 1), np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2 / dims[i + 1])) for i in range(self.num_layers)])
		else:
			raise ValueError("Invalid initialization weight_scale='%s'" % weight_scale)

		# Initialize the biases.
		self.params.update([("b%d" % (i + 1), np.zeros(dims[i + 1])) for i in range(self.num_layers)])

		# Initialization of batch normalization parameters.
		# When using batch norm we need to pass a bn_param dictionary to each layer.
		self.bn_params = []
		if self.normalization == "batchnorm":
			self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
			self.params.update([("gamma%d" % (i + 1), np.ones(hidden_dims[i])) for i in range(self.num_layers - 1)])
			self.params.update([("beta%d" % (i + 1), np.zeros(hidden_dims[i])) for i in range(self.num_layers - 1)])

		# Initialization of dropout parameters.
		# When using dropout we need to pass a dropout_param dictionary to each layer.
		self.dropout_param = {}
		if self.use_dropout:
			self.dropout_param = {"mode": "train", "p": dropout}
			if seed is not None:
				self.dropout_param["seed"] = seed

		# Cast all parameters to the correct datatype.
		for k, v in self.params.items():
			self.params[k] = v.astype(self.dtype)


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
		out = X
		scores = None
		caches = []

		# hidden layers
		for i in range(self.num_layers - 1):
			# affine
			out, cache = affine_forward(out, self.params["W%d" % (i + 1)],
											self.params["b%d" % (i + 1)])
			caches.append(cache)

			# batchnorm
			if self.normalization=='batchnorm':
				out, cache = batchnorm_forward(out, self.params["gamma%d" % (i + 1)],
													self.params["beta%d" % (i + 1)],
													self.bn_params[i])
				caches.append(cache)

			# non-linearity
			out, cache = self.nonlin_forward(out)
			caches.append(cache)

			# dropout
			if self.use_dropout:
				out, cache = dropout_forward(out, self.dropout_param)
				caches.append(cache)

		# output layer
		scores, cache = affine_forward(out, self.params["W%d" % (self.num_layers)],
											self.params["b%d" % (self.num_layers)])
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
		grads = {}
		loss, dscores = cross_entropy_loss(scores, y)

		# output layer
		dout, dW, db = affine_backward(dscores, caches.pop())
		loss += 0.5 * self.reg * np.sum(self.params["W%d" % (self.num_layers)] ** 2)
		dW += self.reg * self.params["W%d" % (self.num_layers)]
		grads["W%d" % (self.num_layers)] = dW
		grads["b%d" % (self.num_layers)] = db

		# hidden layers
		for i in range(self.num_layers - 1, 0, -1):
			# dropout
			if self.use_dropout:
				dout = dropout_backward(dout, caches.pop())

			# non-linearity
			dout = self.nonlin_backward(dout, caches.pop())

			# batchnorm
			if self.normalization=='batchnorm':
				dout, dgamma, dbeta = batchnorm_backward(dout, caches.pop())
				loss += 0.5 * self.reg * np.sum(self.params["gamma%d" % (i)] ** 2)
				dgamma += self.reg * self.params["gamma%d" % (i)]
				grads["gamma%d" % (i)] = dgamma
				grads["beta%d" % (i)] = dbeta

			# affine
			dout, dW, db = affine_backward(dout, caches.pop())
			loss += 0.5 * self.reg * np.sum(self.params["W%d" % (i)] ** 2)
			dW += self.reg * self.params["W%d" % (i)]
			grads["W%d" % (i)] = dW
			grads["b%d" % (i)] = db

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

		# Set train/test mode for batchnorm params and dropout param since they
		# behave differently during training and testing.
		mode = "test" if y is None else "train"
		self.dropout_param["mode"] = mode
		for bn_param in self.bn_params:
			bn_param["mode"] = mode

		# forward pass
		scores, caches = self._forward(X)

		# If test mode, return without computing the backward pass.
		if mode == "test":
			y_pred = np.argmax(scores, axis=1)
			return y_pred

		# backward pass
		loss, grads = self._backward(y, scores, caches)

		return loss, grads