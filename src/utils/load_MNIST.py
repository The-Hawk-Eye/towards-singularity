import numpy as np
import os


def load_data(ROOT):
    """ load all of MNIST """
    train_data = np.loadtxt(ROOT + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt(ROOT + "mnist_test.csv", delimiter=",")

    X_train = np.asfarray(train_data[:, 1:]).reshape((len(train_data), 28, 28))
    y_train = np.array(train_data[:, :1]).astype(int).reshape(len(train_data), )
    X_test = np.asfarray(test_data[:, 1:]).reshape((len(test_data), 28, 28))
    y_test = np.array(test_data[:, :1]).astype(int).reshape(len(test_data), )

    classes = np.array(np.arange(10))

    return X_train, y_train, X_test, y_test, classes


def get_MNIST_data(ROOT, num_training=59000, num_validation=1000, num_test=1000,
                   zero_center=True, normalize=False, whiten=False):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers.
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test, classes = load_data(ROOT)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Zero-center the data: subtract the mean image
    if zero_center:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Normalize the data.
    if normalize:
        std = np.std(X_train, axis=0)
        X_train /= std
        X_val /= std
        X_test /= std

    # Package data into a dictionary.
    data = {"X_train":X_train, "y_train":y_train,
            "X_val":X_val, "y_val":y_val,
            "X_test":X_test, "y_test":y_test,
            "classes":classes}

    return data

#