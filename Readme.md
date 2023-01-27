Instrustions for loading model and generating predictions:

Prepare your test data folder in following way

Test
  -> 0
  -> 1


Run following commands

import pathlib
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
data_dir = pathlib.Path(path to test folder)
test_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
AUTOTUNE = tf.data.experimental.AUTOTUNE
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
new_model = tf.keras.models.load_model('path to model')
predictions = new_model.predict_on_batch(image_batch).flatten()
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)
print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

