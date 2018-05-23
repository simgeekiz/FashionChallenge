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
from keras.preprocessing import image

from PIL import Image

# standard input of exception
DESIRED_IMAGE_SIZE = 290

def image_to_ndarray(path, set_format=True):
    """
    Load a .jpg image.
    
    Arguments:
        path {string} -- file location.
    
    Keyword Arguments:
        pad {set_format} -- True if the image should be edited to have a certain size (default: {True})
    
    Returns:
        ndarray -- image in numpy array.
    """

    img = Image.open(path)
    img.load()
    if set_format:
        return np.asarray(get_right_format(img), dtype='int32')
    else:
        return np.asarray(img, dtype='int32')

def get_right_format(img, desired_size=DESIRED_IMAGE_SIZE, color=(255, 255, 255)):
    """
    Getting the image in the correct format.
    This is done by either downsampling pictures that are too big, or by padding images that are too small.

    Arguments:
        img {obj} -- an image in image format still.

    Keyword Arguments:
        desired_size {int} -- the returned image needs to be a square, this denotes the number of pixels on each side. (default: {DESIRED_IMAGE_SIZE})
        color {(int, int, int)} -- the desired RGB values for padding.

    Returns:
        [obj] -- image in the desired size.
    """

    old_size = img.size

    # create new image in desired size, totally white
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = img.resize(new_size, Image.ANTIALIAS)
    right_format = Image.new("RGB", (desired_size, desired_size), color=color)
    # paste the old image on top of the 'empty' picture created
    right_format.paste(im,
        ((desired_size-new_size[0])//2,
        (desired_size-new_size[1])//2))

    return right_format

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
