import numpy as np
from layers import *

"""
A model for learning word vectors.
When the CBOW algorithm is used, the model aims to predict a center word w_t
given a set of contextual words w_t-j, ..., w_t-1, w_t+1, ... w_t+j, for a 
window size j.
When the Skip-Gram algorithm is used, the model aims to predict a set of
contextual words w_t-j, ..., w_t-1, w_t+1, ... w_t+j given a center word w_t,
for a window size j.
The model is based on a two-layer neural network with no activation function in
the hidden layer. The size of the hidden layer (D) is equal to the size of the
desired subspace for encoding the "word" space.
The weights of the first layer are the word embeddings we are trying to learn.
The weights of the second layer are the weights of a softmax classifier that
performs classification over the entire vocabulary of words.
We train the network with a cross-entropy loss function and L2 regularization on the
weight matrices.
The training is performed using Stochastic Gradient Descent.
"""
class word2vec(object):
    def __init__(self, vocab_size, embed_size,
                 model_type="cbow",
                 negative_sampling=0,
                 word_to_idx={},
                 reg=0.0, dtype=np.float32):
        """
        Inputs:
        - vocab_size: An integer giving the size of the vocabulary V.
        - embed_size: An integer giving the size of the embeddings D.
        - model_type: "cbow" or "skipgram".
        - negative_sampling: An integer giving the number of negative examples to
          be used with negative samling; If 0: standard cross-entropy loss is used.
        - word_to_idx: Dictionary mapping words to integers.
        - reg: An integer giving regularization strength.
        - dtype: numpy datatype to use for computation.
        """
        if model_type not in {"cbow", "skipgram"}:
            raise ValueError("Invalid model_type '%s'" % model_type)
        self.model_type = model_type
        self.negative_sampling = negative_sampling

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.reg = reg
        self.params = {}

        # Initialization of the weights.
        self.params["U"] = np.random.randn(vocab_size, embed_size) / np.sqrt(vocab_size)
        self.params["W"] = np.random.randn(embed_size, vocab_size) / np.sqrt(embed_size)

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def _forward(self, X):
        """
        Input:
        - X: Integer array of shape (N, T) giving indices of words. Each element idx
          of x muxt be in the range 0 <= idx < V.

        Returns:
        - scores: A numpy array of shape (N, T, V) giving scores for each word X[i, j].
        - cache: Values needed for the backward pass.
        """
        N, T = X.shape
        caches = []

        # Compute the word embeddings.
        embeds, cache_embed = word_embedding_forward(X, self.params["U"])
        caches.append(cache_embed)

        # In the cbow algorithm we predict the center word from the window, thus T = 2*window1.
        # In the skip-gram algorithm we predict the window from the center word, thus T = 1.
        V = np.sum(embeds, axis=1) / T

        # Compute the scores.
        scores, cache_out = affine_forward(V, self.params["W"])
        caches.append(cache_out)

        return scores, caches


    def _backward(self, y, scores, caches, window_size):
        """
        Inputs:
        - y: Integer array of labels, of shape (N, T). y[i, j] gives the label for X[i].
        - scores: A numpy array of shape (N, V) giving classification scores.
        - caches: A list of tuples. Each tuple holds cached variables.
        - window_size: Integer, giving the size of the contextual window.

        Returns:
        - loss: A scalar value giving the loss.
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        N, T = y.shape

        # In the cbow algorithm we predict the center word from the window, thus T = 1.
        # In the skip-gram algorithm we predict the window from the center word,
        # thus T = 2 * window_size.
        scores = scores[:, np.newaxis, :]
        scores = np.tile(scores, (1, T, 1))

        # Compute the loss.
        if self.negative_sampling > 0:
            neg_labels = np.random.randint(self.vocab_size, size=(N, self.negative_sampling))
            loss, dscores = negative_sampling_loss(scores.reshape(N*T, -1), y, neg_labels)
        else:
            loss, dscores = cross_entropy_loss(scores.reshape(N*T, -1), y.reshape(N*T))
        dscores = np.sum(dscores.reshape(N, T, -1), axis=1)

        # Compute the gradients of the fully-connected layer.
        dV, dW, _ = affine_backward(dscores, caches.pop())
        loss += 0.5 * self.reg * np.sum(self.params["W"] ** 2)
        dW += self.reg * self.params["W"]

        dV = dV[:, np.newaxis, :]
        if self.model_type == "cbow":
            dV = np.tile(dV, (1, 2 * window_size, 1)) / (2 * window_size)

        # Compute the gradients of the embedding layer.
        dU = word_embedding_backward(dV, caches.pop())
        loss += 0.5 * self.reg * np.sum(self.params["U"] ** 2)
        dU += self.reg * self.params["U"]

        grads = {"W": dW, "U": dU}

        return loss, grads


    def loss(self, X):
        """
        Inputs:
        - X: A numpy array of shape (N, T) of integers giving the input sequence
          to the model. Each element is in the range 0 <= X[i, j] < V.

        Returns:
        - loss: A scalar value giving the loss.
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        N, span = X.shape
        window_size = (span - 1) // 2

        # Split the input into context and target.
        context = np.concatenate((X[:, : window_size], X[:, window_size + 1 : ]), axis=1)
        target = X[:, window_size].reshape(N, 1)

        if self.model_type == "cbow":
            X = context
            y = target
        elif self.model_type == "skipgram":
            X = target
            y = context

        # forward pass
        scores, caches = self._forward(X)

        # backward pass
        loss, grads = self._backward(y, scores, caches, window_size)

        return loss, grads


    def sample(self, input_ids=None, top_k=10):
        """
        Inputs:
        - input_ids: A numpy array of shape (N, ) giving the indices of the words
          for which to find the closest words in the vocabulary.
        - top_k: Int giving the number of top closest words to be returned.

        Returns:
        - similarities: A numpy array of shape (N, top_k) returning the closest
          words for each word.
        """
        if input_ids is None:
            # input_ids = np.random.randint(self.vocab_size, size=(1))
            input_ids = np.array([self.word_to_idx["friend"]])

        N = input_ids.shape[0]
        U = self.params["U"]
        norm_U = U / np.linalg.norm(U, axis=1, keepdims=True)
        similarities = np.dot(norm_U[input_ids], norm_U.T)
        args = np.argsort(similarities, axis=1)[:, ::-1]
        args = args[:, 1:top_k + 1]

        # return args

        similar_words = [[self.idx_to_word[idx] for idx in seq] for seq in args]

        result = []
        for i in range(N):
            txt = " ".join(similar_words[i])
            line = "Nearest to " + self.idx_to_word[input_ids[i]] + ": " + txt
            result.append(line)

        return result