from skimage.io import imread
import tensorflow as tf
import pandas as pd

def mappingfunction(Filename):
    image = tf.readfile(Filename)
    image = tf.image.decode_jpeg(image)
    features = {
        "pedestrian": tf.FixeddLenFeature(image[0:36,0:36])
        "cyclist": tf.FixeddLenFeature(image[0:24,36:60])
        "pedestrianc": tf.FixeddLenFeature(image[24:36,36:48])
        "cyclistc": tf.FixeddLenFeature(image[24:36,48:60])
    }
    parsed_features = tf.parse_single_example(Filename,features)
    return parsed_features["pedestrian"],parsed_features["cyclist",parsed_features["pedestrianc",parsed_features["cyclistc"]


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    dataset = dataset.map(mappingfunction)
    # Return the dataset.
    return dataset

if __name__ = "__main__":
    Dataframe = pd.read_csv("TrainingData_1_test.csv",header=0).drop(["Datum Uhrzeit"])
    Test = train_input_fn(Dataframe.drop(["label"]),Dataframe["label"]])