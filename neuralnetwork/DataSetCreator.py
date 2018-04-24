import tensorflow as tf
import pandas as pd

def mappingfunction(feature,label):
    print(feature['Filename'])
    image = tf.read_file(feature['Filename'])
    image = tf.image.decode_image(image, [36,60])
    features['image'] = image
    return features, label

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    dataset = dataset.map(mappingfunction)
    # Return the dataset.
    return dataset

if __name__ == "__main__":
    Dataframe = pd.read_csv("DummyData.csv",header=0)
    Test = train_input_fn(Dataframe.drop(["label"],axis=1),Dataframe["label"],10)