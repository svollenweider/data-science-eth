import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

imagesize = 15

def cnn_model(features,labels,mode):
    # Extract Images
  
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  
    #Put all images into variables with proper size
    pedestrian = tf.reshape(features['pedestrian'], [-1, 3*imagesize, 3*imagesize, 1])
    cyclist = tf.reshape(features['cyclist'], [-1, 2*imagesize, 2*imagesize, 1])
    pedestrianc = tf.reshape(features['pedestrianc'], [-1, imagesize, imagesize, 1])
    cyclistc = tf.reshape(features['cyclistc'], [-1, imagesize, imagesize, 1])

    List = [pedestrian,cyclist,pedestrainc,cyclistc]
    Sizes = [3,2,1,1]
    
    inputlayer = []
    
    # For each subimage do X to create input layer
    for idx,size,element in enumerate(zip(Sizes,List))
        # Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, input size, input size, 1]
        # Output Tensor Shape: [batch_size, input size, input size, 16]
        conv1 = tf.layers.conv2d(
          inputs=element,
          filters=16,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
          
        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 3
        # Input Tensor Shape: [batch_size,  input size,  input size, 32]
        # Output Tensor Shape: [batch_size,  input size/3,  input size/3, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=3)

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 64]
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 64]
        # Output Tensor Shape: [batch_size, 7, 7, 64]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        pool2_flat = tf.reshape(pool2, [-1,pool2.shape[0]*pool2.shape[1]*pool2.shape[2])
        inputlayer = tf.concat([inputlayer,pool2_flat])
    inputlayer = tf.concat([inputlayer,features['data'])
    

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
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout2, units=7)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)