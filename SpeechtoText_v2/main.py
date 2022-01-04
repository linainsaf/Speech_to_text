# Import libraries
import pickle, gzip
import tensorflow as tf
import numpy as np
from data_preprocessing import encode_single_sample
from model_building import create_network, create_network_with_CNN, CallbackEval, decode_batch_predictions
from jiwer import wer
from tensorflow import keras


# Import data
with gzip.open(('t.pkl.gz'), "rb") as file:
    dtrain, dtest = pickle.load(file, encoding = 'latin-1')

# define dataset
batch_size = 32
# Define the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(dtrain["audio_path"]), list(dtrain["text"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the testing dataset
test_dataset = tf.data.Dataset.from_tensor_slices(
    (list(dtest["audio_path"]), list(dtest["text"]))
)
test_dataset = (
    test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# Get the model
model = create_network(input_dim= 512 // 2 + 1,
                       output_dim= 31)

# Get the model
#model = create_network_with_CNN(input_dim= 512 // 2 + 1,
                       #output_dim= 31)

# Define the number of epochs.
epochs = 5
# Callback function to check transcription on the val set.
validation_callback = CallbackEval(test_dataset)
# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs,
    callbacks=[validation_callback])


# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
# Let's check results on more validation samples
predictions = []
targets = []
for batch in test_dataset:
    X, y = batch
    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)
wer_score = wer(targets, predictions)
print("-" * 100)
print(f"Word Error Rate: {wer_score:.4f}")
print("-" * 100)
for i in np.random.randint(0, len(predictions), 5):
    print(f"Target    : {targets[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 100)