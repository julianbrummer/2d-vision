import tensorflow as tf
from keras import regularizers
import os


# Returns our sequential model with 5 hidden layers with 100 neurons each
# and normalization, dropout after each layer if specified
def create_model(batch_normalization=True, drop_out=True):
    m = tf.keras.models.Sequential([tf.keras.layers.Flatten()])

    for i in range(0, 5):
        m.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)))
        if batch_normalization:
            m.add(tf.keras.layers.BatchNormalization())
        if drop_out:
            m.add(tf.keras.layers.Dropout(0.2))

    m.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    m.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return m


def load_model(checkpoint):
    m = create_model()
    m.load_weights(checkpoint)
    return m


def train_model(m, checkpoint):
    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     # Save weights, every 5-epochs.
                                                     period=5)

    m.fit(x_train, y_train, epochs=100, callbacks=[cp_callback])


# load data set
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# specify checkpoint path
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = create_model()
# train_model(model, checkpoint_path)  # uncomment to train the model yourself

# or just load the model from the latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
loaded_model = load_model(latest)

# evaluate model
loss, acc = loaded_model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Batch normalization and dropout slightly improved the model's accuracy and overfitting, respectively.
# The improvement might be better if we'd also tune their hyperparameters, but I didn't implement that.
