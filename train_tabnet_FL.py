import sys
sys.dont_write_bytecode = True
import os
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from process_data import *
from evaluation import *

def main():

    # -------------------------------------------------------------------
    PATH_TRAIN_LABEL_CSV_FILE = r"unlabel_data/label_train.csv"
    PATH_TRAIN_UNLABEL_CSV_FILE = r"unlabel_data/unlabel_train.csv"
    PATH_TEST_CSV_FILE = r"unlabel_data/test.csv"

    NUM_FEATURES = 14
    OUTPUT_DIMS = 8
    TABNET_PRETRAIN_RATIO = 0.5

    NUM_CLIENTS = 10
    NUM_ROUNDS = 14

    MAX_EPOCHS_LOCAL = 7
    MAX_EPOCHS_SERVER = 100
    # -------------------------------------------------------------------

    # 1. Read dataset
    (X_train_label, y_train_label, list_labels) = Read_Dataset(PATH_TRAIN_LABEL_CSV_FILE)
    X_train_label, X_valid_label, y_train_label, y_valid_label = \
                                        train_test_split(X_train_label, y_train_label, test_size=0.2, random_state=42)
    # (X_train_label, y_train_label) = Handle_ImBalance(X_train_label, y_train_label)
    

    (X_train_unlabel, _) = Read_Dataset(PATH_TRAIN_UNLABEL_CSV_FILE, is_unlabel=True)

    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train_label)
    X_train_label = scaler.transform(X_train_label)
    X_valid_label = scaler.transform(X_valid_label)
    X_train_unlabel = scaler.transform(X_train_unlabel)

    # 2. Split UNLABEL into clients
    list_client_data = np.array_split(X_train_unlabel, NUM_CLIENTS)

    # 3. Define global model (self superviesed TabNet)
    unsupervised_model = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2)
    )

    # Init global model (self-supervised) by labeled dataset in server
    unsupervised_model.fit(
                X_train = X_train_label,        
                eval_set = [X_valid_label],
                max_epochs=MAX_EPOCHS_SERVER,
                pretraining_ratio=TABNET_PRETRAIN_RATIO,
                warm_start = True  
    )

    # 4. Federated Learning process
    list_client_models = []
    for i in range(NUM_CLIENTS):
        client_model = copy.deepcopy(unsupervised_model)
        list_client_models.append(client_model)


    for idx_round in range(NUM_ROUNDS):
        print("---------- [INFO] Round {}".format(idx_round))

        # for name, param in unsupervised_model.network.named_parameters():
        #     print(param)

        # Train client (local) model
        for idx_client in range(NUM_CLIENTS):
            print("----- [INFO] Client {}".format(idx_client))

            # Client get weight from global model
            list_client_models[idx_client].network.load_state_dict(unsupervised_model.network.state_dict())
            
            # Train local model in each user's data
            list_client_models[idx_client].fit(
                X_train = list_client_data[idx_client],
                eval_set = [list_client_data[idx_client]],
                max_epochs=MAX_EPOCHS_LOCAL,
                pretraining_ratio=TABNET_PRETRAIN_RATIO,
                warm_start = True  
            )


        # Compute the average of the local model weights
        unsupervised_model_state = unsupervised_model.network.state_dict()
        for name, param in unsupervised_model.network.named_parameters():

            list_client_layer = []
            for idx_client in range(NUM_CLIENTS):
                current_layer = list_client_models[idx_client].network.state_dict()[name]
                list_client_layer.append(current_layer)

            new_param = torch.mean(torch.stack(list_client_layer), dim=0)
            unsupervised_model_state[name] = new_param

        # Update global (self-supervised) model 
        unsupervised_model.network.load_state_dict(unsupervised_model_state)


    # 6. Fine-tune to supervised TabNet
    print("----- [INFO] Start fine-tuning")
    clf_partial = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9}, 
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
    )

    clf_partial.fit(
        X_train=X_train_label, y_train=y_train_label,
        patience=5,
        eval_set=[(X_valid_label, y_valid_label)],
        eval_name=['valid'],
        eval_metric=[Accuracy],
        from_unsupervised=unsupervised_model,
        max_epochs=MAX_EPOCHS_SERVER
    )

    # 7. Evaluation
    print("----- [INFO] Start evaluation")
    (X_test, y_test, _) = Read_Dataset(PATH_TEST_CSV_FILE)

    X_test = scaler.transform(X_test)

    y_pred = clf_partial.predict(X_test)

    print("X test label: {}".format(X_test.shape))
    print("y test: {}".format(y_test.shape))

    Evaluate_Model_Classifier(y_test, y_pred)


main()
