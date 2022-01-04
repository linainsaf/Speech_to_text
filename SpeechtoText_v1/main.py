import pickle, gzip
import numpy as np
from data_preprocessing import preprocess_single_sample
from data_preprocessing import prepare_for_training
from model_building import create_network

# Import data
with gzip.open(('t.pkl.gz'), "rb") as file:
    dtrain, dtest = pickle.load(file, encoding = 'latin-1')

print(f"Size of the training set: {len(dtrain)}")
print(f"Size of the testing set: {len(dtest)}")

# Preprocess data
spectrograms_train = []
labels_train = []
spectrograms_test = []
labels_test = []
# train samples
for sample in range(len(dtrain)):
    melspec, label = preprocess_single_sample(dtrain.audio_path[sample], dtrain.text[sample])
    spectrograms_train.append(melspec)
    labels_train.append(label)
# test samples
for sample in range(len(dtest)):
    melspec, label = preprocess_single_sample(dtest.audio_path[sample], dtest.text[sample])
    spectrograms_test.append(melspec)
    labels_test.append(label)


# Preprocess data for model
x_train_pad, y_train_pad, x_train_len, y_train_len = prepare_for_training(spectrograms_train, labels_train)
x_test_pad, y_test_pad, x_test_len, y_test_len = prepare_for_training(spectrograms_test, labels_test)


# Define model
nb_labels = 31
nb_features = len(x_train_pad[0][0])
print(nb_features)

model = create_network(nb_features, nb_labels)

# CTC training
model.fit(x=[x_train_pad, y_train_pad, x_train_len, y_train_len], y=np.zeros(len(x_train_pad)),
            batch_size=12, epochs=10)

eval = model.evaluate(x=[x_test_pad,y_test_pad,x_test_len,y_test_len], batch_size=12, metrics=['loss', 'ler', 'ser'])

# predict label sequences
pred = model.predict([x_test_pad, x_test_len], batch_size=12)
for i in range(10):  # print the 10 first predictions
    print("Prediction :", [j for j in pred[i] if j!=-1], " -- Label : ", labels_test[i]) #