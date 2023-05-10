import tensorflow as tf
from modelli import * 
from loss_functions import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
"""
# Creazione del modello
model = create_model()

# Compilazione del modello
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=binary_cross_entropy)

# Addestramento del modello
history = model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(val_X, val_y))

# Valutazione del modello
loss, accuracy = model.evaluate(test_X, test_y)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

    Returns the model with transformer-based architecture
    Args:
        Tx: length of input sequence
        alarm_num_in: number of distinct alarm codes that can appear in an input sequence 
        alarm_num_out: number of labels that are predicted in output
        hparams: a dictionary that contains every parameter used for the model

    Example:
    """
#dictionary with parameters used to build the model
hparams = {
    "hp_embedding_dim": 32,
    "hp_num_heads": 2,
    "hp_ff_dim": 128,
    "hp_nn_dim": 128,
    "dropout_rate": 0.3
        }
Tx = 109 #length of input sequence
alarm_num_in = 154  
alarm_num_out = 556  

#extract xtrain ytrain ...
with open('./processed/MACHINE_TYPE_00_alarms_window_input_1720_window_output_480_offset_60_min_count_20_sigma_3/all_alarms.json', 'rb') as f:
    data = json.load(f)
#print(type(data))


plt.hist(data, bins=10)
# Assume data is a list of dictionaries
all_x_train = []  # Initialize list to hold all x_train values
for sample_data in data.values():
    for key, value in sample_data.items():
        if key == 'x_train':
            all_x_train.append(value)

#plt.hist(all_x_train)
#plt.show()

all_y_train = []  # Initialize list to hold all x_train values
for sample_data in data.values():
    for key, value in sample_data.items():
        if key == 'y_train':
            all_y_train.append(value)

all_x_test = []  # Initialize list to hold all x_train values
for sample_data in data.values():
    for key, value in sample_data.items():
        if key == 'x_test':
            all_x_test.append(value)

all_y_test = []  # Initialize list to hold all x_train values
for sample_data in data.values():
    for key, value in sample_data.items():
        if key == 'y_test':
            all_y_test.append(value)



model = TRM(Tx, alarm_num_in, alarm_num_out, hparams)

alpha = 0.8
gamma = 2.0

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=get_WFL(alpha, gamma))

# define TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)


predictions = model.predict(all_x_train)
#print(all_y_train.shape)
#print(predictions.shape)
'''
for key, value in data.items():
    if key == 'x_train':
        x_train.append(value)
'''
# train the model with callback
model.fit(all_x_train, all_y_train, epochs=100, validation_data=(all_x_test, all_y_test), callbacks=[tensorboard_callback])