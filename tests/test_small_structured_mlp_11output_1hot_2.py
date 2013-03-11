from __future__ import division

import numpy
import cPickle as pkl

from arcade_experiments.pretrained_mlp.structured_mlp.structured_mlp import StructuredMLP, NeuralActivations

from arcade_experiments.pretrained_mlp.structured_mlp.dataset import Dataset
from arcade_experiments.pretrained_mlp.structured_mlp.utils import get_dataset_patches

import theano.tensor as T
import theano

from utils import get_data_patches, get_features

def normalize_data(data):
    """
    Normalize the data with respect to finding the mean and standard deviation of it
    and dividing by mean and standard deviation.
    """
    mu = numpy.mean(data, axis=0)
    sigma = numpy.std(data, axis=0)

    if sigma.nonzero()[0].shape[0] == 0:
        raise Exception("Std dev should not be zero")

    norm_data = (data - mu) / sigma
    return norm_data

def get_binary_labels(labels, no_of_classes=11):
    bin_lbl = []
    for label in labels:
        if label == no_of_classes:
            bin_lbl.append(0)
        else:
            bin_lbl.append(1)
    return numpy.array(bin_lbl)

def load_file(file):
    if file.endswith("pkl"):
        dataset = pkl.load(open(file))
        data = dataset[0]
        labels = dataset[1]
    elif file.endswith("npy"):
        dataset = numpy.load(file)
        data = dataset[0]
        labels = dataset[1]
    elif file.endswith("npz"):
        dataset = numpy.load(file)
        data = dataset["data"]
        labels = dataset["labels"]

    return data, numpy.asarray(labels, dtype="uint8")

def experiment(train_data, train_labels, test_data, test_labels):
    x = T.ftensor3('input_data')
    no_of_patches = 64

    cs_args = {
        "train_args":{
         "learning_rate": 0.1,
         "randomize_mb": True,
         "nepochs": 50,
         "cost_type": "crossentropy",
         "save_exp_data": False,
         "batch_size": 300,
         "enable_dropout": False,
        },
        "test_args":{
         "save_exp_data":False,
         "batch_size": 2000
        }
    }

    post_mlp = StructuredMLP(x,
        in_layer_shape=(no_of_patches, 3, 200, 16),
        layer2_in=64,
        activation=NeuralActivations.Rectifier,
        n_out=1,
        layer1_nout=8,
        quiet=True,
        momentum=0.99,
        save_file="./pkls/structured_mlp_1000_11outs_1hot_small_test.pkl",
        use_adagrad=False)

    post_mlp.set_test_data(test_data, test_labels, patch_mode=False)
    print "=============((((()))))==============="
    print "Training on the dataset."
    post_mlp.train(train_data, train_labels, **cs_args["train_args"])

if __name__=="__main__":
    print "Loading the dataset"

    dir = "/RQexec/gulcehre/datasets/pentomino/onehot/"

    train_file_100k = dir +\
    "1hot_train_data_100k_2.npz"

    test_file_20k = dir +\
    "1hot_test_data_20k_2.npz"

    train_data_100k, train_lbls_100k = load_file(train_file_100k)
    test_data, test_lbls = load_file(test_file_20k)
    print "Started experiment on 10k training dataset with 11 outputs."
    experiment(train_data_100k, train_lbls_100k.flatten(), test_data, test_lbls.flatten())
