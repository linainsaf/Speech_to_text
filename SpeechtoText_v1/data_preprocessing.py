# import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
import librosa


# Define Dictionary of used letters
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to numbers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping numbers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Define properties of our melspectrogram
Nframe = 256
Nhop = 160
Nfft = 512
# Define data directory
base_data_dir = '../../TIMIT/data/'


def preprocess_single_sample(audio_path, transcription):
    #  Audio processing

    # 1. load audio file
    audio, sr = librosa.load(base_data_dir+audio_path)

    # 2. calculate mel-spectrogram
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, win_length=Nframe,
                                          n_fft=Nfft, hop_length=Nhop)

    # 4. Get the spectrogram
    spectrogram = spec.transpose()

    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    ###########################################

    # Process the label
    # 8. Convert label to Lower case
    label = tf.strings.lower(transcription)
    # 9. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 10. Map the characters in label to numbers
    label = char_to_num(label)

    return spectrogram, label


#define parameters
nb_labels = 31 # number of labels (characters)
padding_value = 0 # value for padding input observations

def prepare_for_training(spectrograms, labels, max_padding=800):
    nb_train = len(spectrograms)
    nb_features = len(spectrograms[0][0])

    # pad inputs
    x_train_pad = sequence.pad_sequences(spectrograms, value=float(padding_value), dtype='float32',
                                         padding="post", truncating='post', maxlen=800)
    y_train_pad = sequence.pad_sequences(labels, value=float(nb_labels),
                                         dtype='float32', padding="post")

    # create list of input lengths
    x_train_len = np.asarray([len(x_train_pad[i]) for i in range(nb_train)])
    y_train_len = np.asarray([len(y_train_pad[i]) for i in range(nb_train)])


    return x_train_pad, y_train_pad, x_train_len, y_train_len


