# Deep Learning from scratch

This repository contains implementation of deep neural network architectures written from scratch using only numpy.

The file `layers.py` contains the implementation of simple layers. For each layer a `forward` and a `backward` function are implemented. The `forward` function recieves inputs, weights, and other parameters and returns both an output and a `cache` object storing data needed for the backward pass. The `backward` function recieves upstream derivatives and the `cache` object, and returns gradients with respect to the inputs and weights. These layers are then combined to build classifiers with different architectures.

The file `optimizations.py` contains the implementation of different first-order optimization techniques. Each update rule accepts current weights and the gradient of the loss with respect to those weights and produces the next set of weights.

The following models are implemented:
 * `fc_net.py` implements a fully-connected neural network model
 * `conv_net.py` implements a convolutional neural network model
 * `recurrent_net.py` implements a recurrent neural network model
 * `word2vec.py` implements a model for training word vectors, both Skip-Gram and CBOW are implemented
 * `seq2seq.py` implements a model for sequence-to-sequence learning with attention

For implementing the models the following API is used:
 - learnable parameters are stored in the self.params dictionary which maps parameter names to numpy arrays
 - every model has a `_forward()` method that is used to perform forward propagation through the network and compute the scores
 - every model has a `_backward()` method that is used to perfom backward propagation through the network and compute the gradients

The file `solver.py` contains the implementation of a solver object. A Solver encapsulates all the logic necessary for training models. The Solver performs gradient descent using different update rules defined in `optimizations.py`.