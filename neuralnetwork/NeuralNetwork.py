from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

imagesize = 12

keys = ['richtung', 'Distance', 'MaxFuss', 'MaxVelo', 'Days', 'Uhrzeit', 'Weekday', 'Specialday', 'Lufttemperatur', 'Windgeschwindigkeit', 'Windrichtung', 'Luftdruck', 'Niederschlag', 'Luftfeuchte', 'delayprior']

def mappingfunction(feature,label):
    image = tf.read_file('Images/'+feature['Filename'])
    image = tf.image.decode_jpeg(image,channels=1)
    image = tf.image.convert_image_dtype(image,tf.float32)
    feature['image'] = image
    return feature, label


def input_fn():
    Dataframe = pd.read_csv("TraingDatafinal.csv",header=0)
    features, filenames, labels = Dataframe[keys].values,Dataframe['Filename'].values,Dataframe["label"]
    dataset = tf.data.Dataset.from_tensor_slices(({'x' : features, 'Filename' : filenames},labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(map_func=mappingfunction)
    dataset = dataset.batch(batch_size=100)
    dataset = dataset.prefetch(buffer_size=1000)
    return dataset

def cnn_model(features,labels,mode):
    # Extract Images
  
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]

    #Put all images into variables with proper size
    image = features.pop("image")
    image = tf.reshape(image,[-1,36,60,1])
    pedestrian = image[:,0:36,0:36]
    cyclist = image[:,0:24,36:60]
    pedestrianc = image[:,24:36,36:48]
    cyclistc = image[:,24:36,48:60]

    List = [pedestrian,cyclist,pedestrianc,cyclistc]
    Sizes = [3,2,1,1]
    features['pedestrians'] = pedestrian
    
    # For each subimage do X to create input layer
    inputlayer = None
    for idx,zipped in enumerate(zip(Sizes,List)):
        # Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, input size, input size, 1]
        # Output Tensor Shape: [batch_size, input size, input size, 16]
        size,element = zipped
        conv1 = tf.layers.conv2d(
          inputs=element,
          filters=16,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 3
        # Input Tensor Shape: [batch_size,  input size,  input size, 16]
        # Output Tensor Shape: [batch_size,  input size/3,  input size/3, 16]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=3)

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, input size/3, input size/3, 16]
        # Output Tensor Shape: [batch_size, input size/3, input size/3, 32]
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 7, 7, 32]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        pool2_flat = tf.reshape(pool2, [-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
        if inputlayer is None:
            inputlayer = pool2_flat
        else: inputlayer = tf.concat([inputlayer,pool2_flat],1)
    inputlayer = tf.concat([tf.cast(features['x'],tf.float32),inputlayer],axis=1)
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=inputlayer, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
      
    dense2 = tf.layers.dense(inputs=dropout1, units=100, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1]
    logits = tf.layers.dense(inputs=dropout2, units=1)
    delay = tf.add(logits,0., name="softmax_tensor") 
    predictions = {"delay": delay}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes) if labeled
    #loss for floating output
    def reducedelay(x):
        return tf.subtract(1.,tf.divide(1.,tf.add(1.,tf.square(tf.divide(tf.add(tf.cast(x,tf.float32),60.),180.)))))

    loss = tf.reduce_mean(tf.square(reducedelay(labels) - delay))
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss)

def main(unused_argv):
    # Load training and eval data
    tram2late = tf.estimator.Estimator(model_fn=cnn_model, model_dir="snapshots/")
    tensors_to_log = {"delays": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    tram2late.train(
      input_fn=input_fn,
      hooks=[logging_hook])
    
if __name__ == "__main__": 
    tf.app.run()
 
    
