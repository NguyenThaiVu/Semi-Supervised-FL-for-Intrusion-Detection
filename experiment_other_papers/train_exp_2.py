import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model, clone_model
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical 

from process_data import *
from evaluation import *


def Define_Encoder(input_layer):
    encoded = Dense(150, activation='relu')(input_layer)
    encoded = Dropout(0.1)(encoded)
    encoded = Dense(100, activation='relu')(encoded)
    encoded = Dropout(0.1)(encoded)
    encoded = Dense(50, activation='relu')(encoded)
    return encoded


def Define_Decoder(encoded, num_features):
    decoded = Dense(100, activation='relu')(encoded)
    decoded = Dropout(0.1)(decoded)
    decoded = Dense(150, activation='relu')(decoded)
    decoded = Dropout(0.1)(decoded)
    decoded = Dense(num_features)(decoded) # linear activation
    return decoded


def Define_NN(encoded, output_dims):
    fc = Dense(output_dims, activation='relu')(encoded)
    fc = Dropout(0.01)(fc)
    fc = Dense(output_dims, activation='softmax')(fc)
    return fc



def main():
    """
    This script simulation of paper "Intrusion detection for Softwarized Networks with
                                    Semi-supervised Federated Learning"
    """

    # -------------------------------------------------------------------
    PATH_TRAIN_LABEL_CSV_FILE = r"unlabel_data/label_train.csv"
    PATH_TRAIN_UNLABEL_CSV_FILE = r"unlabel_data/unlabel_train.csv"
    PATH_TEST_CSV_FILE = r"unlabel_data/test.csv"
    
    NUM_FEATURES = 14
    OUTPUT_DIMS = 8

    NUM_CLIENTS = 10
    NUM_ROUNDS = 5

    MAX_EPOCHS_LOCAL = 5
    BATCH_SIZE_LOCAL = 256
    MAX_EPOCHS_SERVER = 20
    BATCH_SIZE_GLOBAL = 256
    # -------------------------------------------------------------------

    # 1. Prepare dataset

    # a. Labeled data
    (X_train_label, y_train_label, list_labels) = Read_Dataset(PATH_TRAIN_LABEL_CSV_FILE)
    X_train_label, X_valid_label, y_train_label, y_valid_label = \
                                        train_test_split(X_train_label, y_train_label, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train_label)
    X_train_label = scaler.transform(X_train_label)
    X_valid_label = scaler.transform(X_valid_label)

    y_train_label = to_categorical(y_train_label, num_classes=OUTPUT_DIMS)
    y_valid_label = to_categorical(y_valid_label, num_classes=OUTPUT_DIMS)

   
    # b. Unlabel data
    (X_train_unlabel, _) = Read_Dataset(PATH_TRAIN_UNLABEL_CSV_FILE, is_unlabel=True)
    X_train_unlabel, X_valid_unlabel = train_test_split(X_train_unlabel, test_size=0.2, random_state=42)
    
    X_train_unlabel = scaler.transform(X_train_unlabel)
    X_valid_unlabel = scaler.transform(X_valid_unlabel)

    # 2. Split UNLABEL into clients
    list_client_data = np.array_split(X_train_unlabel, NUM_CLIENTS)

    # 3. Define global AE
    input_layer = Input(shape=(NUM_FEATURES,))

    encoded = Define_Encoder(input_layer)
    decoded = Define_Decoder(encoded, NUM_FEATURES)
    AE_model = Model(inputs=input_layer, outputs=decoded)

    AE_model.compile(optimizer='adam', loss='mean_squared_error')

    # AE_model.fit(X_train_unlabel, X_train_unlabel,
    #             epochs=MAX_EPOCHS_LOCAL,
    #             batch_size=BATCH_SIZE_LOCAL,
    #             shuffle=True,
    #             validation_data=(X_valid_unlabel, X_valid_unlabel))


    # 4. FL process
    # a. Create list of clients local model
    list_client_models = []
    for i in range(NUM_CLIENTS):
        client_model = clone_model(AE_model)    # Define AE
        client_model.compile(optimizer='adam', loss='mean_squared_error')
        list_client_models.append(client_model)


    # b. Train FL
    for idx_round in range(NUM_ROUNDS):
        print("---------- [INFO] Round {}".format(idx_round))

        for idx_client in range(NUM_CLIENTS):
            print("----- [INFO] Client {}".format(idx_client))
            # Get weight from global model
            list_client_models[idx_client].set_weights(AE_model.get_weights())

            # Train local model in each user's data
            list_client_models[idx_client].fit(list_client_data[idx_client], list_client_data[idx_client],\
                                                epochs=MAX_EPOCHS_LOCAL, batch_size=BATCH_SIZE_LOCAL)

        # Compute FedAvg AE
        average_weights = []
        for i in range(len(AE_model.get_weights())):
            layer_weights = np.array([list_client_models[j].get_weights()[i]  for j in range(NUM_CLIENTS)])
            average_layer_weights = np.mean(layer_weights, axis=0)
            average_weights.append(average_layer_weights)

        # Update the global model weights with the average of the local model weights
        AE_model.set_weights(average_weights)


    # 5. Fine-tuning global NN with labeled data
    fc = Define_NN(encoded, OUTPUT_DIMS)
    prediction_model = Model(inputs=input_layer, outputs=fc)
    prediction_model.summary()

    prediction_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    prediction_model.fit(X_train_label, y_train_label,\
            validation_data=(X_valid_label, y_valid_label),\
            epochs=MAX_EPOCHS_SERVER, batch_size=BATCH_SIZE_GLOBAL,\
            callbacks=[early_stop])


    # 6. Evaluation
    print("----- [INFO] Start evaluation")

    (X_test, y_test, _) = Read_Dataset(PATH_TEST_CSV_FILE)
    X_test = scaler.transform(X_test)

    y_pred = prediction_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    print("X test shape: {}".format(X_test.shape))
    print("y test shape: {}".format(y_test.shape))
    print("y predict shape: {}".format(y_pred.shape))
    print("y test: {}".format(dict(Counter(y_test))))

    Evaluate_Model_Classifier(y_test, y_pred)


main()