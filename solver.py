import numpy as np
import optimizations


"""
A Supervised Solver encapsulates all the logic necessary for training supervised models.
To train a model construct a SupervisedSolver instance and pass the model, the dataset and
other options (e.g. learning rate, batch size) to the constructor. Then call the train() method
to run the optimization procedure. After the train() method returns, model.params will
contain the parameters that performed best on the validation set over the course of training.
In addition, the instance variable solver.loss_history will contain a list of all losses
encountered during training and the instance variables solver.train_acc_history and
solver.val_acc_history will be lists of the accuracies of the model on the training and
validation set at each epoch.

A Solver works on a model object and a dataset object that must conform to the following API:
- model.params must be a dictionary mapping string parameter names to numpy arrays containing
  parameter values.
- model.loss(X, y) must be a function that computes training-time loss and gradients,
  and test-time predictions.
- dataset.train_batch must be a function that generates a batch of training examples.
- dataset.len returns the size of the training set.

The Solver performs stochastic gradient descent using different update rules defined in
"optimizations.py".
"""
class SupervisedSolver(object):
	def __init__(self, model, dataset, **kwargs):
		"""
		Required arguments:
		- model: A model object conforming to the API described above.
		- dataset: A dataset object conforming to the API described above.

		Optional arguments:
		- update_rule: A string giving the name of an update rule. Default is "sgd".
		- optim_config: A dictionary containing hyperparameters that will be
		  passed to the chosen update rule. Each update rule requires different
		  hyperparameters (see optim.py) but all update rules require a
		  "learning_rate" parameter so that should always be present.
		- lr_decay: A scalar for exponentially decaying the learning rate.
		- batch_size: Size of minibatches used to compute loss and gradient during training.
		- num_epochs: The number of epochs to run for during training.
		- print_every: Integer. Model accuracy will be calculated every print_every iterations.
		- verbose: Bool; if set to false then no output will be printed during training.
		"""
		self.model = model
		self.dataset = dataset

		# Unpack keyword arguments.
		self.update_rule = kwargs.pop("update_rule", "sgd")
		self.optim_config = kwargs.pop("optim_config", {})
		self.lr_decay = kwargs.pop("lr_decay", 1.0)
		self.batch_size = kwargs.pop("batch_size", 128)
		self.num_epochs = kwargs.pop("num_epochs", 10)
		self.print_every = kwargs.pop("print_every", 10)
		self.verbose = kwargs.pop("verbose", True)

		# Throw an error if there are extra keyword arguments.
		if len(kwargs) > 0:
			extra = ', '.join("'%s'" % k for k in list(kwargs.keys()))
			raise ValueError("Unrecognized arguments %s" % extra)

		# Make sure the update rule exists, then replace the string
		# name with the actual function.
		if not hasattr(optimizations, self.update_rule):
			raise ValueError("Invalid update_rule '%s'" % self.update_rule)
		self.update_rule = getattr(optimizations, self.update_rule)

		self._reset()


	def _reset(self):
		"""
		Set up some book-keeping variables for optimization.
		This function should not be called manually.
		"""
		# Set up some variables for book-keeping.
		self.best_val_acc = 0
		self.best_params = {}
		self.loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []

		# Make a deep copy of the optim_config for each parameter.
		# Each parameter keeps track of its variables needed to perform an update rule
		# (e.g. velocities, caches, etc.).
		self.optim_configs = {}
		for p in self.model.params:
			d = {k: v for k, v in self.optim_config.items()}
			self.optim_configs[p] = d


	def _step(self):
		"""
		Make a single gradient update.
		This function is called by train() and should not be called manually.
		"""
		# Make a mini-batch of training data.
		X_batch, y_batch = self.dataset.train_batch(self.batch_size)

		# Compute the loss and the gradients.
		loss, grads = self.model.loss(X_batch, y_batch)
		self.loss_history.append(loss)

		# Perform a parameter update.
		for p, w in self.model.params.items():
			dw = grads[p]
			config = self.optim_configs[p]
			next_w, next_config = self.update_rule(w, dw, config)
			self.model.params[p] = next_w
			self.optim_configs[p] = next_config


	def check_accuracy(self, X, y):
		"""
		Inputs:
		- X: A numpy array of training examples.
		- y: A numpy array of training labels.

		Returns:
		- acc: A scalar giving the fraction of instances that were correctly predicted.
		"""
		# Compute predictions in batches.
		# Split X and y into batches of this size to avoid using too much memory.
		N = X.shape[0]
		num_batches = N // self.batch_size
		if N % self.batch_size != 0:
			num_batches += 1
		y_pred = []
		for i in range(num_batches):
			start = i * self.batch_size
			end = (i + 1) * self.batch_size
			pred = self.model.loss(X[start:end])
			y_pred.append(pred)
		y_pred = np.hstack(y_pred)
		acc = np.mean(y_pred == y)

		return acc


	def train(self):
		"""
		Run optimization to train the model.
		"""
		num_train = self.dataset.num_train()
		iterations_per_epoch = max(num_train // self.batch_size, 1)
		num_iterations = self.num_epochs * iterations_per_epoch

		if self.verbose:
			print("Number of iterations per epoch: %d" % iterations_per_epoch)

		epoch = 0
		for t in range(num_iterations):
			# Perform a signle gradient update.
			self._step()

			# At every print_every iteration print training loss.
			if self.verbose and t % self.print_every == 0:
				print("(Iteration %d / %d) loss: %.5f" %
								(t, num_iterations, self.loss_history[-1]))

			# At the end of every epoch do some work.
			epoch_end = (t % iterations_per_epoch == 0) or (t == num_iterations - 1)
			if epoch_end:
				# Increment the epoch counter.
				epoch += 1

				# Check train and val accuracy.
				num_val = self.dataset.num_val()
				X_b, y_b = self.dataset.train_batch(num_val)
				X_v, y_v = self.dataset.val_batch()
				train_acc = self.check_accuracy(X_b, y_b)
				val_acc = self.check_accuracy(X_v, y_v)
				self.train_acc_history.append(train_acc)
				self.val_acc_history.append(val_acc)

				# Maybe print train and val accuracy.
				if self.verbose:
					print("(Epoch %d / %d) train acc: %f; val_acc: %f" %
								(epoch - 1, self.num_epochs, train_acc, val_acc))

				# Decay the learning rate.
				for k in self.optim_configs:
					self.optim_configs[k]["learning_rate"] *= self.lr_decay

				# Keep track of the best model.
				if val_acc > self.best_val_acc:
					self.best_val_acc = val_acc
					self.best_params = {}
					# self.best_params.update([(k, v.copy) for k, v in self.model.params.items()])
					for k, v in self.model.params.items():
						self.best_params[k] = v.copy()

		# At the end of training swap the best params into the model
		self.model.params = self.best_params


"""
An Unsupervised Solver encapsulates all the logic necessary for training unsupervised models.
To train a model construct a UnsupervisedSolver instance and pass the model, the dataset and
other options (e.g. learning rate, batch size) to the constructor. Then call the train() method
to run the optimization procedure. After the train() method returns, model.params will
contain the parameters of the model arrived at after the optimizations.

In addition, the instance variable solver.loss_history will contain a list of all losses
encountered during training.

A Solver works on a model object and a dataset object that must conform to the following API:
- model.params must be a dictionary mapping string parameter names to numpy arrays containing
  parameter values.
- model.loss(X) must be a function that computes training-time loss and gradients,
  and test-time predictions.
- dataset.train_batch must be a function that generates a batch of training examples.
- dataset.num_train must be a function returns the size of the training set.

The Solver performs stochastic gradient descent using different update rules defined in
"optimizations.py".
"""
class UnsupervisedSolver(object):
	def __init__(self, model, dataset, **kwargs):
		"""
		Required arguments:
		- model: A model object conforming to the API described above.
		- dataset: A dataset object conforming to the API described above.

		Optional arguments:
		- update_rule: A string giving the name of an update rule. Default is "sgd".
		- optim_config: A dictionary containing hyperparameters that will be
		  passed to the chosen update rule. Each update rule requires different
		  hyperparameters (see optim.py) but all update rules require a
		  "learning_rate" parameter so that should always be present.
		- lr_decay: A scalar for exponentially decaying the learning rate.
		- batch_size: Size of minibatches used to compute loss and gradient during training.
		- num_epochs: The number of epochs to run for during training.
		- print_every: Integer. Results will be printed every print_every iterations.
		- verbose: Bool; if set to false then no output will be printed during training.
		"""
		self.model = model
		self.dataset = dataset

		# Unpack keyword arguments.
		self.update_rule = kwargs.pop("update_rule", "sgd")
		self.optim_config = kwargs.pop("optim_config", {})
		self.lr_decay = kwargs.pop("lr_decay", 1.0)
		self.batch_size = kwargs.pop("batch_size", 128)
		self.num_epochs = kwargs.pop("num_epochs", 10)
		self.print_every = kwargs.pop("print_every", 10)
		self.verbose = kwargs.pop("verbose", True)

		# Throw an error if there are extra keyword arguments.
		if len(kwargs) > 0:
			extra = ', '.join("'%s'" % k for k in list(kwargs.keys()))
			raise ValueError("Unrecognized arguments %s" % extra)

		# Make sure the update rule exists, then replace the string
		# name with the actual function.
		if not hasattr(optimizations, self.update_rule):
			raise ValueError("Invalid update_rule '%s'" % self.update_rule)
		self.update_rule = getattr(optimizations, self.update_rule)

		self._reset()


	def _reset(self):
		"""
		Set up some book-keeping variables for optimization.
		This function should not be called manually.
		"""
		# Set up some variables for book-keeping.
		self.loss_history = []

		# Make a deep copy of the optim_config for each parameter.
		# Each parameter keeps track of its variables needed to perform an update rule
		# (e.g. velocities, caches, etc.).
		self.optim_configs = {}
		for p in self.model.params:
			d = {k: v for k, v in self.optim_config.items()}
			self.optim_configs[p] = d


	def _step(self):
		"""
		Make a single gradient update.
		This function is called by train() and should not be called manually.
		"""
		# Make a mini-batch of training data.
		X_batch = self.dataset.train_batch(self.batch_size)

		# Compute the loss and the gradients.
		loss, grads = self.model.loss(X_batch)
		self.loss_history.append(loss)

		# Perform a parameter update.
		for p, w in self.model.params.items():
			dw = grads[p]
			config = self.optim_configs[p]
			next_w, next_config = self.update_rule(w, dw, config)
			self.model.params[p] = next_w
			self.optim_configs[p] = next_config


	def train(self):
		"""
		Run optimization to train the model.
		"""
		num_train = self.dataset.num_train()
		iterations_per_epoch = max(num_train // self.batch_size, 1)
		num_iterations = self.num_epochs * iterations_per_epoch

		if self.verbose:
			print("Number of iterations per epoch: %d" % iterations_per_epoch)

		epoch = -1
		for t in range(num_iterations):
			# Perform a signle gradient update.
			self._step()

			# At the end of every epoch do some work.
			epoch_end = (t % iterations_per_epoch == 0) or (t == num_iterations - 1)
			if epoch_end:
				# Increment the epoch counter.
				epoch += 1

				# Decay the learning rate.
				for k in self.optim_configs:
					self.optim_configs[k]["learning_rate"] *= self.lr_decay

			# At every print_every iteration print some results.
			if self.verbose and t % self.print_every == 0:
				print("(Iteration %d / %d); Epoch(%d / %d); loss: %.5f" %
								(t, num_iterations, epoch - 1, self.num_epochs, self.loss_history[-1]))
				print("Sample:\n", self.model.sample()[0])