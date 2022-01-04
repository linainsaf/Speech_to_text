from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Activation, Input, Bidirectional
from tensorflow.keras.optimizers import SGD
from CTC import CTCModel


def create_network(nb_features, nb_labels):

    # Define the network architecture
    # Input Layer
    input_layer = Input(name='input', shape=(None, nb_features))
    # LSTM layers
    x = Bidirectional(LSTM(500, return_sequences=True))(input_layer)
    x = layers.Dropout(rate=0.5)(x)
    x = Bidirectional(LSTM(500, return_sequences=True))(x)
    x = layers.Dropout(rate=0.5)(x)
    x = Bidirectional(LSTM(500, return_sequences=True))(x)
    x = layers.Dropout(rate=0.5)(x)
    x = Bidirectional(LSTM(500, return_sequences=True))(x)
    x = layers.Dropout(rate=0.5)(x)
    x = Bidirectional(LSTM(500, return_sequences=True))(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    x = TimeDistributed(Dense(nb_labels + 1, name="dense"))(x)
    output_layer = Activation("sigmoid")(x)
    # Model
    model = CTCModel([input_layer], [output_layer])
    # Compile the model and return
    model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9))
    return model