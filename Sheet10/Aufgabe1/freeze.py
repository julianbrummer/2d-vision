"""
Set the first argument to the name of the model file you want to load.
If no model file is specified a new network is created.
"""
from __future__ import division
from sklearn.model_selection import KFold
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import sys, os
import time
from itertools import compress

millis = int(round(time.time() * 1000))

# create dir to save model
model_dir = "model"
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# load model
model_file = None
if len(sys.argv) == 2:
    model_file = os.path.join(model_dir, sys.argv[1]+".meta")
    if not os.path.isfile(model_file):
        raise Exception("Model file: {0} not found!".format(model_file))

n_input = 784       # MNIST data input (img shape 28*28)
n_classes = 10      # MNIST total classes (0-9 digits)

# Hyperparameters
n_layer = 5             # number of hidden layer
n_neurons = 100         # number of neurons/nodes
training_epochs = 1     # because of the crossvalidation this will actually run more often
learning_rate = 0.001
batch_size = 20
display_step = 1
beta = 0.01             # beta value for L2 regularization

# create main session instance
sess = tf.Session()

# restore a model if necessary
if model_file:
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, model_file[:-5])

#input
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

# init weights and bias
weights = {}
biases = {}
n_prev = n_input

# restore or init weights and biases
xavier = tf.contrib.layers.xavier_initializer()

for i in range(n_layer):
    weights["h"+str(i)] = tf.get_variable("w"+str(i),
                                          shape=[n_prev, n_neurons],
                                          initializer=xavier,
                                          trainable=False)
    biases["b"+str(i)] = tf.get_variable("b"+str(i),
                                         shape=[n_neurons],
                                         initializer=xavier,
                                         trainable=False)
    n_prev = n_neurons

# add weights and bias for output layer
weights["out"] = tf.get_variable("out_w", shape=[n_prev, n_classes], initializer=xavier, trainable=False)
biases["out"] = tf.get_variable("out_b", shape=[n_classes], initializer=xavier, trainable=False)


def multilayer_perceptron(layer):
    """
    Create the layers of a MLP.
    """
    # create hidden layer
    for i in range(n_layer):
        # or one could use a dense layer
        layer = tf.add(tf.matmul(layer, weights["h"+str(i)]), biases["b"+str(i)])
        # add batch normalization before non linearity, so that the model converges faster
        # this might lower the accuracy of our MLP
        layer = tf.layers.batch_normalization(layer)
        # activation function
        layer = tf.nn.relu(layer, name="layer"+str(i))
    # add a dropout layer before the fully connected output layer to help reduce overfitting
    layer = tf.layers.dropout(layer, rate=0.5)
    # return output layer
    return tf.matmul(layer, weights["out"]) + biases["out"]


def cross_validate(session, split_size=5):
    """
    Calculate crossvalidation.
    This is really not a good idea for large datasets. 
    """
    results = []
    kf = KFold(n_splits=split_size)
    for train_idx, val_idx in kf.split(mnist.train.images,  mnist.train.labels):
        train_x = mnist.train.images[train_idx]
        train_y = mnist.train.labels[train_idx]
        val_x = mnist.train.images[val_idx]
        val_y = mnist.train.labels[val_idx]
        run_train(session, train_x, train_y)
        results.append(session.run(accuracy, feed_dict={X: val_x, Y: val_y}))
    return results


def initialize_uninitialized_vars(sess):
    """
    Initialize all uninitialized variables.
    """
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
       sess.run(tf.variables_initializer(not_initialized_vars))


# create multilayer perceptron
mlp = multilayer_perceptron(X)

# define loss and optimizer to minimze gradient
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mlp, labels=Y))
# add L2 regularization to weights
loss_op += tf.add_n([tf.nn.l2_loss(v) for k, v in weights.items()]) * beta
# define adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")
train_op = optimizer.minimize(loss_op, name="train_op")

# calculate accuracy
pred = tf.nn.softmax(mlp)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
initialize_uninitialized_vars(sess)

if not model_file:
    saver = tf.train.Saver()


def run_train(session, train_x, train_y):
    """Train the MLP"""
    for epoch in range(training_epochs):
        avg_loss = 0.0
        # number of batches
        total_batch = int(train_x.shape[0]/batch_size)

        # split out dataset into batches
        for i in range(total_batch):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            batch_y = train_y[i*batch_size:(i+1)*batch_size]
            # run our trainings session
            _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            avg_loss += loss / total_batch

        # display
        if epoch % display_step == 0:
            print("Epoch: {0} loss={1:.9f}".format(epoch+1, avg_loss))
            # save a checkpoint each epoche
            saver.save(sess, "./model/model.ckpt", global_step=(epoch+1)*total_batch)

result = cross_validate(sess)
print("Cross-validation: {0}".format(result))
print("Accuracy: {0}".format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))
millis = int(round(time.time() * 1000)) - millis
print(millis)
