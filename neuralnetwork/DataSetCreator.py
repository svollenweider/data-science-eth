import tensorflow as tf
import pandas as pd
import numpy


def mappingfunction(feature,label):
    print(feature['Filename'])
    image = tf.read_file(feature['Filename'])
    image = tf.image.decode_image(image)
    feature['image'] = image
    return feature, label


def train_input_fn(features, labels, batchsize):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.map(mappingfunction)
    dataset = dataset.shuffle(1000).repeat().batch(100)
    #
    # Return the dataset
    return dataset.make_one_shot_iterator().get_next()
