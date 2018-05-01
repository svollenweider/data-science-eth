from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

imagesize = 12
nooffilters = 32 
nstrides = 2
firstlayer = 2048
secondlayer = 1024
#thirdlayer = 512

keys = ['richtung', 'Distance', 'MaxFuss', 'MaxVelo', 'Days', 'Uhrzeit', 'Weekday', 'Specialday', 'Lufttemperatur', 'Windgeschwindigkeit', 'Windrichtung', 'Luftdruck', 'Niederschlag', 'Luftfeuchte', 'delayprior']

def mappingfunction(feature,label):
    image = tf.read_file('Images/'+feature['Filename'])
    image = tf.image.decode_jpeg(image,channels=1)
    image = tf.image.convert_image_dtype(image,tf.float32)
    feature['image'] = image
    return feature, label

def masking(feature, label):
    mask = tf.not_equal(label,0)
    randomfloats = tf.random_uniform(mask.shape)
    addmask = tf.less(randomfloats,0.2)
    mask = tf.logical_or(mask,addmask)
    return mask
    
    
 
def input_fn():
    Dataframe = pd.read_csv("TrainingDatafinal.csv",header=0)
    features, filenames, labels = Dataframe[keys].values,Dataframe['Filename'].values,Dataframe["altlabel"]
    dataset = tf.data.Dataset.from_tensor_slices(({'x' : features, 'Filename' : filenames},labels))
    #mask = dataset.map(map_func=masking)
    #dataset = tf.boolean_mask(dataset,mask)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=mappingfunction)
    dataset = dataset.batch(batch_size=100)
    dataset = dataset.prefetch(buffer_size=1000)
    return dataset
    
def input_fn_eval():
    Dataframe = pd.read_csv("evalDatafinal.csv",header=0)
    features, filenames, labels = Dataframe[keys].values,Dataframe['Filename'].values,Dataframe["altlabel"]
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
          filters=nooffilters,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 3
        # Input Tensor Shape: [batch_size,  input size,  input size, 16]
        # Output Tensor Shape: [batch_size,  input size/3,  input size/3, 16]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=nstrides)

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, input size/3, input size/3, 16]
        # Output Tensor Shape: [batch_size, input size/3, input size/3, 32]
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=nooffilters*2,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 7, 7, 32]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=nstrides)
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        pool2_flat = tf.reshape(pool2, [-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
        if inputlayer is None:
            inputlayer = pool2_flat
        else: inputlayer = tf.concat([inputlayer,pool2_flat],1)
    inputlayer = tf.concat([tf.cast(features['x'],tf.float32),inputlayer],axis=1)
    #inputlayer = tf.cast(features['x'],tf.float32)
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=inputlayer, units=firstlayer, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
      
    dense2 = tf.layers.dense(inputs=dropout1, units=secondlayer, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
      
    #dense3 = tf.layers.dense(inputs=dropout2, units=thirdlayer, activation=tf.nn.relu)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 7]
    output = tf.layers.dense(inputs=dropout2, units=7)
    #loss for floating output
    '''
    def reducedelay(x):
        return tf.subtract(1.,tf.divide(1.,tf.add(1.,tf.square(tf.divide(tf.add(tf.cast(x,tf.float32),60.),180.)))))
    
    def ireducedelay(y):
        y = tf.cast(y,tf.float32)
        return tf.cast(tf.divide(tf.add(tf.multiply(-60.,y),60-tf.sqrt(tf.multiply(180.**2,tf.multiply((1.-y),y)))),y-1),tf.int64)
    '''
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes) if labeled
    
    delay = tf.identity(output, name="delay")
    labels = tf.identity(labels, name = "label")
    

       
    def maplabelstoprob(label):
        gamma = 1
        cd = lambda x: 1/(np.pi*gamma)*tf.divide(gamma**2,tf.add(tf.square(x-tf.transpose([tf.cast(label,tf.float32)])),gamma**2))
        return cd(tf.cast(tf.range(7),tf.float32))
    
 
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=maplabelstoprob(labels), logits=output))
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output)
    #loss = tf.square(tf.cast(labels-predictions['classes'],tf.float32))
    #loss = tf.reduce_mean(tf.sqrt(tf.square(labels - tf.cast(delay, tf.float32))))
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=output, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
    } 
    
    logs = tf.concat([tf.expand_dims(predictions['classes'],1),tf.expand_dims(labels,1)],1,name="Accuracy")
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric = {}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric)

def main(unused_argv):
    # Load training and eval data
    tram2late = tf.estimator.Estimator(model_fn=cnn_model, model_dir="snapshots/")
    tensors_to_log = { "logging": "Accuracy"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    tram2late.train(
      input_fn=input_fn,
      hooks=[logging_hook])
    
    eval_results = tram2late.evaluate(input_fn=input_fn_eval)
    print(eval_results)
if __name__ == "__main__":
    tf.app.run()
 
    
