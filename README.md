# End to End Speech recognition using RNN and CTC 

This repository contains two folders speechtotext v1 and speechtotext v2. the first folder contains the code for training a CTC model based on the model here : https://github.com/cyprienruffino/CTCModel
The second contains the training of a Keras LSTM model with the CTC Loss function.

## How the code works 
Both folders contain 3 essential functions : 
- data_preprocessing.py : Handels the preprocessing of the data, including transformation of .wav files to spetrograms and text to numbers.  
- model_building.py : Here, we build our model
- main.py : For training our model, and for making predictions.


First we start by installing the requirements. To do so we run this line on our terminal ( in the folder ) : 
```shell
pip install -r requirements.txt 
```

then we run main.py 
