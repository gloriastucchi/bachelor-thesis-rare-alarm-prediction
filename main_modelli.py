import os
import tensorflow as tf
import modelli as m
import loss_functions as lf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
from sklearn.metrics import f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer


def run_model():

    #! Loads and init JSON data


    #! Loads the model from a file in the saved_model directory

    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the saved model directory
    saved_model_dir = os.path.join(current_dir, 'saved_model')

    model_found = False
    try: 
        # Load the saved model
        loaded_model = tf.saved_model.load(saved_model_dir)
        #! loaded_model = tf.keras.models.load_model(saved_model_dir)
        model_found = True
    
    except FileNotFoundError:
        print(f"Error: Saved model directory {saved_model_dir} not found.")
    
    except OSError as e:
        print (f"Error loading saved model: {e}")

    except ValueError as e:
        print (f"Error loading saved model: {e}")

    # Check for a valid model 
    if (model_found == False):

        #! Model not found: generate it 
        answer = input("Do you want to generate a model ? (this might take some time) (y/n)")

        if answer.lower() == "y": 
            print("Starting with model generation..")

        else: 
            print("Program is going to terminate") 

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
    alarm_num_out = 154  # ! numero totale di possibili allarmi, prima erano 10 e dava errore di non match of dimensions

    #load data from json produced in previous code
    with open('./processed/MACHINE_TYPE_00_alarms_window_input_1720_window_output_480_offset_60_min_count_20_sigma_3/all_alarms.json', 'rb') as f:
        data = json.load(f)

    # Assume data is a list of dictionaries
    all_x_train = []  # Initialize list to hold all x_train values
    for sample_data in data.values():
        for key, value in sample_data.items():
            if key == 'x_train':
                all_x_train.append(value)

    # Size of all_x_train
    all_x_train_size = len(all_x_train)
    print("Size of all_x_train:", all_x_train_size)

    # Size of the dictionaries in all_x_train
    dict_size = len(all_x_train[0])
    print("Size of dictionaries in all_x_train:", dict_size)

    #plt.hist(all_x_train)
    #plt.show()

    all_y_train = []  # Initialize list to hold all x_train values
    for sample_data in data.values():
        for key, value in sample_data.items():
            if key == 'y_train':
                all_y_train.append(value)
    all_y_train = np.array(all_y_train)
    all_y_train = all_y_train[:, :154]

    # Size of all_x_train
    all_y_train_size = len(all_x_train)
    print(f"Size of all_y_train: {all_y_train_size}")

    # Size of the dictionaries in all_x_train
    dict_size = len(all_y_train[0])
    print(f"Size of dictionaries in all_y_train 1: {dict_size}")
    dict_size = len(all_y_train[1])
    print(f"Size of dictionaries in all_y_train 2: {dict_size}")


    #! Test Data 

    # Initialize list to hold all x_test values
    all_x_test = []
    for sample_data in data.values():
        for key, value in sample_data.items():
            if key == 'x_test':
                all_x_test.append(value)

    # Initialize list to hold all y_test values
    all_y_test = []  
    for sample_data in data.values():
        for key, value in sample_data.items():
            if key == 'y_test':
                all_y_test.append(value)
    all_y_test = np.array(all_y_test)
    all_y_test = all_y_test[:, :154]

    
    all_x_test_size = len(all_x_test)
    print("Size of all_x_test:", all_x_test_size)
    all_y_test_size = len(all_y_test)
    print("Size of all_y_test:", all_y_test_size)
    dict_size = len(all_y_test[1])
    print(f"Size of dictionaries in all_y_test 2: {dict_size}")

    all_x_train = np.array(all_x_train)
    all_x_test = np.array(all_x_test)
    num_classes = 41054
    #all_y_train_categorical = np.eye(num_classes)[all_y_train]
    # Determine all possible classes
    all_classes = np.arange(154)  # Replace `total_num_classes` with the actual number of classes

    # Create an instance of MultiLabelBinarizer and specify all possible classes
    mlb = MultiLabelBinarizer(classes=all_classes)

    # Fit and transform the labels
    all_y_train_encoded = mlb.fit_transform(all_y_train)
    all_y_test_encoded = mlb.transform(all_y_test)

    # Print the shape of the transformed labels
    print("Shape of all_y_train_encoded:", all_y_train_encoded.shape)
    print("Shape of all_y_test_encoded:", all_y_test_encoded.shape)

    #all_y_test_categorical = to_categorical(all_y_test, num_classes=154)

    #create the model with TRM
    model = m.TRM(Tx, alarm_num_in, alarm_num_out, hparams)

    #define parameters to be passed to the loss function
    alpha = 0.8
    gamma = 2.0
    model.summary()
    #! loss=lf.get_WFL(alpha, gamma)
    #model is compiled
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=lf.get_WFL(alpha, gamma))

    # define TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

    
    #print(all_y_train.shape)
    #print(predictions.shape)


    # train the model (loss obtained close to 3.5)
    # ! epoch era 100, cambiata per far prima
    model.fit(all_x_train, all_y_train_encoded, epochs=10, callbacks=[tensorboard_callback])
    """
    # Get the absolute path of the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the saved model directory
    saved_model_dir = os.path.join(current_dir, 'saved_model')

    # Save the model in SavedModel format
    tf.saved_model.save(model, saved_model_dir)

    """
    #! Loads the model from a file in the saved_model directory

    # Get the absolute path of the current directory
    #current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the saved model directory
    #saved_model_dir = os.path.join(current_dir, 'saved_model')

    # Load the saved model
    #loaded_model = tf.saved_model.load(saved_model_dir)

    #! Prediction
    
    # Make predictions using the model
    predictions = model.predict(all_x_test)
    """
    # Get the prediction function from the loaded model's signature
    # predict_fn = model.signatures["serving_default"]

    # Convert input data to TensorFlow tensors of dtype float32
    # all_x_test_tf = tf.convert_to_tensor(all_x_test, dtype=tf.float32)

    #predictions = model(all_x_test_tf)

    # Make predictions using the loaded model
    # predictions = predict_fn(all_x_test_tf)
    
    #TODO: prediction key dense 7 
    # Print the keys of the predictions dictionary
    #print(f"Prediction keys: {list(predictions.keys())}")

    # Convert the predictions to a numpy array
    predictions_np = predictions["dense_7"].numpy()

    # Print the predicted probabilities
    print("Predicted probabilities:")
    print(predictions_np)
    """

    # Set the threshold
    threshold = 0.5

    # Create an empty binary predictions array
    binary_predictions = np.zeros_like(predictions)

    # Find the indices of classes with prediction probabilities above the threshold
    class_indices = np.where(predictions > threshold)

    # Set the corresponding indices in the binary predictions array to 1
    binary_predictions[class_indices] = 1
    # Calculate the binary prediction
    #binary_predictions = (predictions_np > threshold)

    # Convert the necessary variables to numpy arrays
    #all_y_test = np.array(all_y_test)
    #binary_predictions = np.array(binary_predictions)

    #! F1 Scores

    # Calculate F1 scores to evaluate the model's performance
    f1_scores = []

    # Iterate over each label to calculate f1 scores
    for label in range(all_y_test.shape[1]):
        # Extract the true labels for the current label
        y_true_label = all_y_test[:, label]
        
        # Extract the predicted labels for the current label
        y_pred_label = binary_predictions[:, label]
        
        # Ensure that the number of samples is the same
        min_samples = min(len(y_true_label), len(y_pred_label))
        y_true_label = y_true_label[:min_samples]
        y_pred_label = y_pred_label[:min_samples]
   
    num_classes = len(y_true_label[0])
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)

    for true_label, predicted_label in zip(all_y_test, binary_predictions):
        for i in range(num_classes):
            if true_label[i] == 1 and predicted_label[i] == 1:
                true_positives[i] += 1
            elif true_label[i] == 0 and predicted_label[i] == 1:
                false_positives[i] += 1
            elif true_label[i] == 1 and predicted_label[i] == 0:
                false_negatives[i] += 1

    precisions = true_positives / (true_positives + false_positives)
    recalls = true_positives / (true_positives + false_negatives)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    average_f1_score = np.mean(f1_scores)


    #! Output communication

    # Print the f1_macro result on the test data
    print(f"The obtained f1 macro is : {f1_macro}")

    '''
    # Example: Predict using the loaded model
    predictions = loaded_model.predict(all_x_test)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    f1_scores = f1_score(all_y_test, binary_predictions, average=None)
    f1_micro = f1_score(all_y_test, binary_predictions, average='micro')
    '''

if __name__ == "__main__":
    run_model()     