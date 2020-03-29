# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:31:06 2020

@author: Varad Srivastava
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

# 1. Data importing function
def input_fn(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

# 2. Define the inputs
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.train)[0]},
    y=input(mnist.train)[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)


eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.test)[0]},
    y=input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)

# 3. Define Feature Columns
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]


# 4. Instantiate an estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.optimizers.Adam(1e-4),
    n_classes=10,
    model_dir="./tmp/mnist_model"
)

# 5. Train the model
classifier.train(input_fn=train_input_fn, steps=100)

# 6. Evaluate the trained model
eval_result = classifier.evaluate(input_fn=eval_input_fn)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

