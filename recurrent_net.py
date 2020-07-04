import numpy as np
from layers import *


"""
A recurrent neural network model for processing sequential inputs.
The network has a hidden dimension of size H and operates on inputs of size D.
The network operates on minibatches of size N.
The network works on sequences of length T and calculates the hidden states
h_1, h_2, ..., h_T.
The nonlinearity performed by the network is given by "cell_type".
The initial hidden state h0 can be learned as a parameter of the network or it can
be assumed to be 0.
"""
class RecurrentNetwork(object):
	def __init__(self, input_dim, hidden_dim, output_dim,
				 n_layers=1, weight_scale=1e-3, reg=0.0,
				 cell_type="rnn",
				 dtype=np.float32):
		"""
		Inputs:
		- input_dim: Dimension D of the embeddings.
		- hidden_dim: Dimension H for the hidden state of the RNN.
		- n_layers: Number of layers for the RNN.
		- reg: Scalar giving L2 regularization strength.
		- cell_type: What type of cell non-linearity to use: "rnn" or "lstm".
		- dtype: Numpy datatype to use for computation.
		"""
		if cell_type not in {"rnn", "lstm"}:
			raise ValueError("Invalid cell_type '%s'" % cell_type)

		self.cell_type = cell_type
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.n_layers = n_layers
		self.reg = reg
		self.dtype = dtype
		self.params = {}

		# Initialize the weights and biases.
		dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
		for i in range(n_layers):
			self.params["Wx_%d" % i] = np.random.randn(hidden_dim, dim_mul * hidden_dim) / np.sqrt(hidden_dim) * weight_scale
			self.params["Wh_%d" % i] = np.random.randn(hidden_dim, dim_mul * hidden_dim) / np.sqrt(hidden_dim) * weight_scale
			self.params["b_%d" % i] = np.zeros(dim_mul * hidden_dim)
		self.params["Wx_0"] = np.random.randn(input_dim, dim_mul * hidden_dim) / np.sqrt(input_dim) * weight_scale

		self.params["W_out"] = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim) * weight_scale
		self.params["b_out"] = np.zeros(output_dim)

		# Cast all parameters to the correct datatype.
		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)


	def _forward(self, X, h0=None):
		"""
		Input:
		- X: A numpy array of input data of shape (N, T, D)
		- h0: A numpy array of shape (N, H) giving the initial hidden state.

		Returns:
		- scores: A numpy array of shape (N, T, M) giving the output scores of the recurrent network.
		- caches: Values needed for the backward pass.
		"""
		N, T, D = X.shape
		H = self.hidden_dim

		caches = []

		if h0 is None:
			h0 = np.zeros((N, H))

		h = X
		for i in range(self.n_layers):
			if self.cell_type == "rnn":
				h, cache = recurrent_forward(h, h0, self.params["Wx_%d" % i], self.params["Wh_%d" % i],
											 self.params["b_%d" % i])
			elif self.cell_type == "lstm":
				h, cache = lstm_forward(h, h0, self.params["Wx_%d" % i], self.params["Wh_%d" % i],
										self.params["b_%d" % i])
			caches.append(cache)

		scores, out_cache = temporal_affine_forward(h, self.params["W_out"], self.params["b_out"])
		caches.append(out_cache)

		return scores, caches


	def _backward(self, y, scores, caches):
		"""
		Inputs:
		- y: A numpy array of shape (N, T, M) giving output data.
		- scores: Upstream derivatives of the scores of shape (N, T, M).
		- caches: Cached variables for the backward pass.

		Returns:
		- loss: A scalar value giving the loss.
		- grads: Dictionary with the same keys as self.params, mapping parameter
		  names to gradients of the loss with respect to those parameters.
		"""
		grads = {}

		loss, dscores = temporal_cross_entropy_loss(scores, y)
		dh, dW_out, db_out = temporal_affine_backward(dscores, caches.pop())
		loss += 0.5 * self.reg * np.sum(self.params["W_out"])
		dW_out += self.reg * self.params["W_out"]

		grads["W_out"] = dW_out
		grads["b_out"] = db_out

		for i in range(self.n_layers -1, -1, -1):
			if self.cell_type == "rnn":
				dh, dh0, dWx, dWh, db = recurrent_backward(dh, caches.pop())
			elif self.cell_type == "lstm":
				dh, dh0, dWx, dWh, db = lstm_backward(dh, caches.pop())

			loss += 0.5 * self.reg * (np.sum(self.params["Wx_%d" % i]) + np.sum(self.params["Wh_%d" % i]))
			dWx += self.reg * self.params["Wx_%d" % i]
			dWh += self.reg * self.params["Wh_%d" % i]

			grads["Wx_%d" % i] = dWx
			grads["Wh_%d" % i] = dWh
			grads["b_%d" % i] = db

		return loss, grads


	def loss(self, X, y, h0=None):
		"""
		Inputs:
		- X: A numpy array of shape (N, T, D) giving input data.
		- y: A numpy array of shape (N, T, M) giving output data.
		- h0: A numpy array of shape (N, D) giving the initial hidden state.

		Returns:
		- loss: A scalar value giving the loss.
		- grads: Dictionary with the same keys as self.params, mapping parameter
		  names to gradients of the loss with respect to those parameters.
		"""
		# Cast the input data to the same datatype as the parameters of the model.
		X = X.astype(self.dtype)

		# forward pass
		scores, caches = self._forward(X, h0)

		# backward pass
		loss, grads = self._backward(y, scores, caches)

		return loss, grads


	def sample(self, start, embed, max_length=30):
		"""
		Inputs:
		- start: A numpy array of shape (N, D) giving input data.
		- embed: Function used to generate embedding from index.
		- max_length: Maximum length T of generated outputs.

		Returns:
		- outputs: Array of shape (N, max_length, M) giving sampled outputs.
		"""
		N, D = start.shape
		H = self.hidden_dim

		# Cast the input data to the same datatype as the parameters of the model.
		start = start.astype(self.dtype)

		x = start
		prev_h = np.zeros((N, self.n_layers, H))
		next_h = np.zeros((N, self.n_layers, H))
		prev_c = np.zeros((N, self.n_layers, H))
		next_c = np.zeros((N, self.n_layers, H))

		# Store the generated sequence.
		sequence = np.zeros((N, max_length, self.output_dim))
		sequence[:, 0, :] = x
		for t in range(1, max_length):
			# Make one time-step.
			prev_input = x
			for i in range(self.n_layers):
				if self.cell_type == "rnn":
					h, _ = recurrent_step_forward(prev_input, prev_h[:, i, :].reshape(N, H),
												  self.params["Wx_%d" % i], self.params["Wh_%d" % i],
												  self.params["b_%d" % i])
				elif self.cell_type == "lstm":
					h, c, _ = lstm_step_forward(prev_input, prev_h[:, i, :].reshape(N, H), prev_c[:, i, :],
												self.params["Wx_%d" % i], self.params["Wh_%d" % i],
												self.params["b_%d" % i])
				prev_input = h
				next_h[:, i, :] = h
				next_c[:, i, :] = c

			# Compute the output and use it as next input.
			scores, _ = affine_forward(h, self.params["W_out"], self.params["b_out"])
			out = np.argmax(scores, axis=1)
			x = embed(out, self.input_dim)

			# Store the output of the last layer.
			sequence[:, t, :] = x

			# Update the hidden and cell states.
			prev_h = next_h
			prev_c = next_c

		return sequence