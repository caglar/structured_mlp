import numpy as np
import math

import theano

#Get patches from a single image according to patch_size
def get_patches(img, patch_size=(8, 8)):
    if img.ndim == 1:
        dim = math.sqrt(img.shape[0])
        img = img.reshape((dim, dim))

    img_rows = img.shape[0]
    img_cols = img.shape[1]
    patches = None

    for i in xrange(img_rows / patch_size[0]):
        for j in xrange(img_cols / patch_size[1]):
            patch = img[i * patch_size[0]: (i + 1)* patch_size[0], j * patch_size[0]: (j + 1) * patch_size[1]]
            patch = patch.flatten()
            if patches is None:
                patches = patch
            else:
                if patches.ndim != patch.ndim:
                    patches = np.vstack((patches, [patch]))
                else:
                    patches = np.vstack(([patches], [patch]))
    return patches

def shared_dataset(data_x, name="x"):
    shared_x = theano.shared(np.asarray(data_x.tolist(), dtype=theano.config.floatX))
    shared_x.name = name
    return shared_x

#Get patches from the dataset for each image
def get_dataset_patches(dataset, patch_size=(8, 8)):
    dataset_patches = []
    for data in dataset:
        dataset_patches.append(get_patches(data, patch_size))
    return np.array(dataset_patches)

#Get the three object patches.
def get_three_obj_patches(img, img_pre, patch_size):
    img_patches = get_patches(img, patch_size)
    img_3patches = []
    img_3pres = []
    for i in xrange(img_patches.shape[0]):
        if img_pre[i] != 0 and img_pre[i] != -1:
            img_3pres.append(np.asarray(img_pre[i].tolist(), dtype=theano.config.floatX))
            img_3patches.append(np.asarray(img_patches[i].tolist(),
            dtype=theano.config.floatX))

    return np.array(img_3patches), np.array(img_3pres)

#Get patches from the dataset for each image
def get_dataset_obj_patches(dataset, dataset_pre, patch_size=(8, 8)):

    dataset_patches = None
    dataset_pres = []

    for i in xrange(dataset.shape[0]):
        img3_patches, img3_pres = get_three_obj_patches(dataset[i], dataset_pre[i], patch_size)
        if img3_patches is not None and len(img3_patches) >= 1:
            if dataset_patches is None:
                dataset_patches = img3_patches
            else:
                dataset_patches = np.vstack((dataset_patches, img3_patches))
            dataset_pres.append(img3_pres.tolist())

    return dataset_patches, np.asarray(dataset_pres.tolist(),
    dtype=theano.config.floatX)
