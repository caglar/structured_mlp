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
        data = pkl.load(open(file))
    elif file.endswith("npy"):
        data=numpy.load(file)
    return data[0], numpy.asarray(data[1], dtype="uint8")

def experiment(train_data, train_labels, test_data, test_labels):
    x = T.ftensor3('input_data')
    no_of_patches = 64
    train_patches = get_dataset_patches(train_data)

    cs_args = {
        "train_args":{
         "learning_rate": 0.002,
         "nepochs": 60,
         "cost_type": "crossentropy",
         "save_exp_data": False,
         "batch_size": 100,
         "enable_dropout": False
        },
        "test_args":{
         "save_exp_data":False,
         "batch_size": 2000
        }
    }

    post_mlp = StructuredMLP(x,
        in_layer_shape=(no_of_patches, no_of_patches, 200, 256),
        layer2_in=2048,
        layer1_nout=50,
        activation=NeuralActivations.Rectifier,
        n_out=1,
        quiet=True,
        save_file="structured_mlp_100k_50out_g.pkl",
        use_adagrad=False)

    post_mlp.set_test_data(test_data, test_labels)
    print "=============((((()))))==============="
    print "Training on the dataset."
    post_mlp.train(train_patches, train_labels, **cs_args["train_args"])

if __name__=="__main__":
    print "Loading the dataset"

    dir = "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/"

    train_file_100k = dir +\
    "pento64x64_100k_seed_98313722_64patches.npy"

    test_file_20k = dir +\
    "pento64x64_20k_64patches_seed_112168712_64patches.npy"

    train_data_100k, train_lbls_100k = load_file(train_file_100k)
    test_data, test_lbls = load_file(test_file_20k)

    print "Started experiment on 100k training dataset with 50 output units in the first layer."

    experiment(train_data_100k, get_binary_labels(train_lbls_100k.flatten(), 10), test_data, get_binary_labels(test_lbls.flatten(), 10))
