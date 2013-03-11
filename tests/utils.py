import numpy as np
import math

import theano
from dataset import Dataset

def normalize(data):
    diff = np.abs(np.min(data, axis=0))
    data = data + diff
    scaler = np.max(data, axis=0)
    data = (data)/scaler
    return data

def standardize(data):
    """
    Normalize the data with respect to finding the mean and standard deviation of it
    and dividing by mean and standard deviation.
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)

    if sigma.nonzero()[0].shape[0] == 0:
        raise Exception("Std dev should not be zero")

    norm_data = (data - mu) / sigma
    return norm_data

def as_floatX(variable):
    """
       This code is taken from pylearn2:
       Casts a given variable into dtype config.floatX
       numpy ndarrays will remain numpy ndarrays
       python floats will become 0-D ndarrays
       all other types will be treated as theano tensors
    """

    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)

def safe_update(dict_to, dict_from):
    """
    Like dict_to.update(dict_from), except don't overwrite any keys.
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to

def get_data_patches(data):
    ds = Dataset(is_binary=True)
    ds.setup_pretraining_dataset(data_dict[data_key],
    train_split_scale=1.0,
    patch_size=(8, 8))

    ds_test = Dataset(is_binary=True)
    ds_test.setup_pretraining_dataset(data_dict["ds_test"],
    train_split_scale=1.0,
    patch_size=(8, 8))

    XTrain, YTrain = ds.Xtrain_patches, ds.Ytrain
    XTest, YTest = ds_test.Xtrain_patches, ds_test.Ytrain
    return XTrain, YTrain, XTest, YTest

def get_features(ae,
    data,
    n_hiddens):
    feats_enc = np.zeros((data.shape[0], data.shape[1], n_hiddens), dtype=theano.config.floatX)
    print "creating features"
    for i in xrange(data.shape[1]):
        h = ae.encode(x_in=data[:, i])
        h_fn = theano.function([], h)
        feats = h_fn()
        feats_enc[:, i] = feats
    return feats_enc

