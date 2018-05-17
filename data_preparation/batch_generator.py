"""
Module to generate batches for both the training and validation set.
For training, use the BatchGenerator.
For validation, use the BatchSequence.
"""

from os import listdir
from os.path import join

from math import ceil, floor
import numpy as np

from keras.utils import Sequence
from PIL import Image

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600

def image_to_ndarray(path, pad=True):
    """
    Load a .jpg image.
    
    Arguments:
        path {string} -- file location.
    
    Keyword Arguments:
        pad {bool} -- True if the image should be padded (default: {True})
    
    Returns:
        ndarray -- image in numpy array.
    """

    img = Image.open(path)
    img.load()
    if pad:
        return pad_image(np.asarray(img, dtype='int32'))
    else:
        return np.asarray(img, dtype='int32')

def pad_image(img, target_width=IMAGE_WIDTH, target_height=IMAGE_HEIGHT, mode='constant', constant_values=255):
    """
    Padding a numpy image in 3D. The padding is done along x- and y-axis.
    The third channel will never be padded as it is suppossed to contain the RGB-values.
    
    Arguments:
        img {ndarray} -- an image in numpy array format.
    
    Keyword Arguments:
        target_width {int} -- the desired width of the returned image (default: {IMAGE_WIDTH})
        target_height {int} -- the desired height of the returned image (default: {IMAGE_HEIGHT})
        mode {str} -- whichever mode is preferred; see https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html (default: {'constant'})
        constant_values {int} -- if mode is constant this depicts the value of the padded pixels (default: {255}) white.
    
    Returns:
        ndarray -- padded image in the prescribed format.
    """

    img_width = img.shape[0]
    img_height = img.shape[1]
    
    # Calculate how many padding is needed on the axis.
    h_padding = target_width - img_width
    v_padding = target_height - img_height
    
    return np.pad(img, (
        (ceil(h_padding/2), floor(h_padding/2)), # x-axis
        (ceil(v_padding/2), floor(v_padding/2)), # y-axis
        (0,0) # nothing as this is the color channel
    ), mode=mode, constant_values=constant_values)

class BatchGenerator(object):
    """
    This class generates batches that can be provided to a neural network.
    It can be used for training only. For validation use the BatchSequence class.
    """

    def __init__(self, input_dir, y, batch_size, augmentation_fn=None):
        """
        Constructor of the BatchGenerator.
        
        Arguments:
            input_dir {string} -- directory in which the images are stored.
            y {[rows=indices, cols=labels]} -- labels corresponding to the images in input_dir, in multilabel notation.
            batch_size {int} -- expected size of the generated batches.

        Keyword Arguments:
            augmentation_fn {function} -- augmentor function for the data (default: {None})
        """
        self.input_dir = input_dir
        self.x = listdir(input_dir)
        self.y = y
        self.batch_size = batch_size  # number of patches per batch
        self.augmentation_fn = augmentation_fn  # augmentation function

    def __iter__(self):
        """
        Make the object iterable.

        Returns:
            self.
        """
        return self

    def __next__(self):
        """
        Next iteration.

        Returns:
            function -- builds a mini-batch.
        """
        return self.next()

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        Returns:
            int -- the number of batches possible such that every sample of the class with the least samples is seen once.
        """
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def next(self):
        """
        Build a mini-batch.

        Returns:
            (ndarray, ndarray) -- a batch with training samples and a batch with the corresponding labels.
        """
        # pick random values from the training set
        idxs = np.random.randint(0, len(self.x), self.batch_size)

        batch_x = [self.x[i] for i in idxs]
        batch_y = [self.y[i] for i in idxs]

        return np.array([
            image_to_ndarray(join(self.input_dir, x))
                for x in batch_x]), np.array(batch_y)

class BatchSequence(Sequence):
    """
    This class generates batches that can be provided to a neural network.
    It can be used for validation only. For training use the BatchGenerator class.
    
    Arguments:
        Sequence {class} -- a sequence never repeats items.
    """

    def __init__(self, input_dir, y, batch_size):
        """
        Constructor of the BatchSequence.

        Arguments:
            input_dir {string} -- directory in which the images are stored.
            y {[rows=indices, cols=labels]} -- labels corresponding to the images in input_dir, in multilabel notation.
            batch_size {int} -- expected size of the generated batches.
        """

        self.input_dir = input_dir
        self.x = listdir(input_dir)  # path to patches in glob format
        self.y = y
        self.batch_size = batch_size  # number of patches per batch

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        Returns:
            int -- the number of batches possible such that every sample of the class with the least samples is seen once.
        """
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        """
        Get the next batch from the validation set. Since it is a sequence, it will never give records twice.
        
        Arguments:
            idx {int} -- offset
        
        Returns:
            (ndarray, ndarray) -- a batch with validation samples and a batch with the corresponding labels.
        """

        # create indices
        idx_min = idx * self.batch_size
        # make sure to never go out of bounds
        idx_max = np.min([idx_min + self.batch_size, len(self.x)])
        idxs = np.arange(idx_min, idx_max)
        
        batch_x = [self.x[i] for i in idxs]
        batch_y = [self.y[i] for i in idxs]
        
        return np.array([
            image_to_ndarray(join(self.input_dir, x))
                for x in batch_x]), np.array(batch_y)
