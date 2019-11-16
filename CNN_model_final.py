import os
import csv
'''
Created in collaboration with Stuart Reid and Elan van Biljon.
Convolutional Neural Network created by Schirrmeister et al (2017) - Deep Learning with Convolutional Neural Networks 
for EEG Decoding and Visualisation
'''

import numpy as np
import pandas as pd
import numpy.random as rng
from sklearn.preprocessing import MinMaxScaler
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D 
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import model_from_json

import tensorflow.keras.backend as K

import tensorflow as tf

import matplotlib.pyplot as plt

class PrepareDataset(object):

    def __init__(self, raw_csv_data_dir, 
                 train_test_split_method="individual", 
                 train_test_split_pct=0.7,
                 prediction_time_step=0.5, 
                 *args, **kwargs):
        """
        TODO define what this method does

        :param raw_csv_data_dir: this is the folder where the data lives.
        :param prediction_time_steps: these are the frame sizes we want to look at (% of the total window - helps with variable window sizes).
        :param train_test_split_method: this is the methodology we want to use for splitting out train and test.
        :param train_test_split_pct: this is how much data goes into the training set (%)
        """
        self.raw_csv_data_dir = raw_csv_data_dir

        if train_test_split_method not in ["individual", "grouped", "separated"]:
            raise AttributeError(
                "train_test_split_method should be either"\
                "'grouped', or 'separated'")

        self.train_test_split_method = train_test_split_method
        self.train_test_split_pct = train_test_split_pct
        self.prediction_time_step = prediction_time_step

        self.data_transformers = {}
        self.raw_data, self.ml_data = {}, {}

    def read_data_from_raw_csv_data_dir(self):
        """
        TODO define what this method does
        This function reads over the data files and converts the csvs into pandas.
        It uses the file name to access the "decision" which becomes the "response/ label"

        """

        data = {}

        for csv_file in os.listdir(self.raw_csv_data_dir): 
            if not csv_file.endswith('.csv'):
                continue

            # Load the CSV file using pandas
            csv_data = pd.read_csv(os.path.join(
                self.raw_csv_data_dir, csv_file), header=None)

            # From the name extract relevant information
            person, decision, _ = csv_file.split("_")

            if person not in data:
                # Add the person to our in memory dataset
                data[person] = {"left": [], "right": []}

            if decision == "101":
                # Add this window to the left decisions list
                data[person]["left"].append(csv_data.values)
            else:
                # Add this window to the right decisions list
                data[person]["right"].append(csv_data.values)

            # print("loading data from {}".format(csv_file))

        self.raw_data = data

    def split_raw_data_into_frames(self):
        """
        TODO define what this method does
        This takes the epoch / window and determines which of it will be used for that particular frame.
        The frames are used for the data; to give the designation. This will allow us to see if we can out put 
            high enough classification accuracies using the earlier frames. 
        Frames are based on a percentage of the entire window / epoch to account for varying lengths of epochs read in as csv files. 
        Varying epoch lengths are only appropriate in the case of the "separated" analysis as 
            we don't have to concatenate different lengths of data.
        """
        for person, data in self.raw_data.items():

            for i, lft_datum in enumerate(data["left"]):
                _, cols = lft_datum.shape
                cols_subset = int(cols * self.prediction_time_step)
                cols_subset = list(range(cols_subset))
                lft_datum_subset = lft_datum[:, cols_subset]
                self.raw_data[person]["left"][i] = lft_datum_subset

            for i, rgt_datum in enumerate(data["right"]):
                _, cols = rgt_datum.shape
                cols_subset = int(cols * self.prediction_time_step)
                cols_subset = list(range(cols_subset))
                rgt_datum_subset = rgt_datum[:, cols_subset]
                self.raw_data[person]["right"][i] = rgt_datum_subset

        return None

    def print_shape_of_dataset(self):
        """
        TODO define what this method does
        This prints out what the previous function has done to the data in terms of the "frame" size etc.
        """

        print("RAW DATA SHAPE:")

        for person, data in self.raw_data.items():
            print("\nDATA FOR PERSON {}".format(person))

            print("\tLEFT DECISIONS: {}".format(len(data["left"])))
            for i, lft_datum in enumerate(data["left"]):
                print("\t\t {} rows {} cols {}".format(i, *lft_datum.shape))
            
            print("\tRIGHT DECISIONS: {}".format(len(data["right"])))
            for i, rgt_datum in enumerate(data["right"]):
                print("\t\t {} rows {} cols {}".format(i, *rgt_datum.shape))

        print("ML DATA SHAPE:")

        for person, data in self.ml_data.items():
            print("\n\tML DATA FOR PERSON {}".format(person))

            print("\t\t patterns:", self.ml_data[person]["patterns"].shape)
            print("\t\t responses:", self.ml_data[person]["responses"].shape)

    def convert_to_ml_dataset(self):
        """
        TODO explain what this method does
        TODO change the Standard Scaler to MinMaxScaler
        """
        ml_dataset = {}
        for person, data in self.raw_data.items():

            if person not in ml_dataset:
                ml_dataset[person] = {"patterns": [], "responses": []}
                self.data_transformers[person] = {"scalers": []}

            for i, lft_datum in enumerate(data["left"]):
                # Fit transform for this datum / frame
                # sc_i = StandardScaler().fit(lft_datum)                
                sc_i = MinMaxScaler().fit(lft_datum) 
                
                # Save the transformation object for reporting
                self.data_transformers[person]["scalers"].append(sc_i)

                # Add the transformed datum & response to the ml dataset
                ml_dataset[person]["patterns"].append(sc_i.transform(lft_datum))
                ml_dataset[person]["responses"].append([-1])

            for i, rgt_datum in enumerate(data["right"]):
                # Fit transform for this datum / frame
                #sc_i = StandardScaler().fit(rgt_datum)                
                sc_i =MinMaxScaler().fit(rgt_datum) 
                # Save the transformation object for reporting
                self.data_transformers[person]["scalers"].append(sc_i)

                # Add the transformed datum & response to the ml dataset
                ml_dataset[person]["patterns"].append(sc_i.transform(lft_datum))
                ml_dataset[person]["responses"].append([1])

            # Convert this to a numpy array for use inside the model
            ml_dataset[person]["patterns"] = np.array(ml_dataset[person]["patterns"])
            ml_dataset[person]["responses"] = np.array(ml_dataset[person]["responses"])

        self.ml_data = ml_dataset
        np.savez("cached_ml_dataset", ml_dataset)

    def split_ml_dataset(self):
        """
        TODO define what this method does
        This method splits the data. 
        Grouped = concatenate all the sets, then split them. This will result in 'mixed' train and test sets. Mixed in the sense
             that we train on some; test on some. These windows need to be of equal size to be concatenated. 
        Separated = train on some and test on others. 
        """
        if len(self.raw_data.keys()) == 1:
            print("Defaulting back to grouped splitting because only 1 person")
            self.train_test_split_method = "grouped"

        if self.train_test_split_method == "grouped":

            people = list(self.ml_data.keys())
            
            patterns = [self.ml_data[p]["patterns"] for p in people]
            responses = [self.ml_data[p]["responses"] for p in people]

            patterns = np.vstack(patterns)
            responses = np.vstack(responses)

            # shuffle data to mitigate catastrophic forget
            num_patters = patterns.shape[0]
            indices = np.arange(num_patters)
            # np.random.seed(0)
            np.random.shuffle(indices)

            train_i_stop = int(num_patters*self.train_test_split_pct)
            train_i = indices[:train_i_stop]
            
            num_val = int((num_patters - train_i_stop) / 2)
            val_i_stop = num_val + train_i_stop
            validation_i = indices[train_i_stop:val_i_stop]

            test_i = indices[val_i_stop:]

            train_patterns = patterns[train_i]
            validation_patterns = patterns[validation_i]
            test_patterns = patterns[test_i]

            train_responses = responses[train_i]
            validation_responses = responses[validation_i]
            test_responses = responses[test_i]

            # tot_num_pairs = patterns.shape[0]
            # all_pairs = list(range(tot_num_pairs))

            # num_train = int(tot_num_pairs * self.train_test_split_pct)
            # train_ixs = rng.choice(all_pairs, num_train, replace=False)

            # test_val_ixs = [i for i in all_pairs if i not in train_ixs] #This is for all the other ones not 
            #                                                         #included in the separation in the previous step(line above)
            # ## stop indent
            # num_test_and_val = int(np.floor(tot_num_pairs - num_train))
            # num_test = int(np.floor(num_test_and_val / 2))
            # num_val = num_test_and_val - num_test
            # test_ixs = rng.choice(test_val_ixs, num_test, replace=False)
            # validation_ixs = [i for i in test_val_ixs if i not in test_ixs] #This is for all the other ones not 
            # #included in the separation in the previous step(line above)

            # # test_ixs = [i for i in all_pairs if i not in train_ixs] #This is for all the other ones not 
            # #                                                         #included in the separation in the previous step(line above)

            # train_patterns = patterns[train_ixs, :, :]
            # test_patterns = patterns[test_ixs, :, :]
            # validation_patterns = patterns[validation_ixs, :, :]

            # train_responses = responses[train_ixs, :]
            # test_responses = responses[test_ixs, :]
            # validation_responses = responses[validation_ixs, :]

            return train_patterns, train_responses, test_patterns, test_responses, validation_patterns, validation_responses

        elif self.train_test_split_method == "separated":

            people = list(self.ml_data.keys())
            num_people = len(people)
            num_train = int(num_people * self.train_test_split_pct)

            people_train = rng.choice(people, num_train, replace=False)
            people_test_val_ixs =  [p for p in people if p not in people_train]
            # people_test = [p for p in people if p not in people_train]

            people_test_val_num = int(np.floor(num_people - num_train))
            num_people_test = int(np.floor(people_test_val_num / 2))
            num_val_people = people_test_val_num - num_people_test

            print("Total number of people in dataset:", num_people)
            print("Number of people in training set:", num_train)
            print("Number of people in test set:",num_people_test)
            print("Number of people in validation set:", num_val_people)
            people_test = rng.choice(people_test_val_ixs, num_people_test, replace=False)
            people_validation = [p for p in people_test_val_ixs if p not in people_test]
            
            train_patterns = [self.ml_data[p]["patterns"] for p in people_train]
            train_responses = [self.ml_data[p]["responses"] for p in people_train]

            train_patterns = np.vstack(train_patterns)
            train_responses = np.vstack(train_responses)

            test_patterns = [self.ml_data[p]["patterns"] for p in people_test]
            test_responses = [self.ml_data[p]["responses"] for p in people_test]

            test_patterns = np.vstack(test_patterns)
            test_responses = np.vstack(test_responses)

            validation_patterns = [self.ml_data[p]["patterns"] for p in people_validation]
            validation_responses = [self.ml_data[p]["responses"] for p in people_validation]

            validation_patterns = np.vstack(validation_patterns)
            validation_responses = np.vstack(validation_responses)
            

            return train_patterns, train_responses, test_patterns, test_responses, validation_patterns, validation_responses

def create_model(input_X0shape):
    shape = (None, input_X0shape)

    model = Sequential()
    model.add(Conv2D(25, (10), padding = "same", data_format = 'channels_last', input_shape = input_X0shape, kernel_constraint = max_norm(2., axis=(0,1,2))))
    
    #model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (1,3), strides = (1,3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(50, (10), padding = "same", data_format = 'channels_last',kernel_constraint = max_norm(2., axis=(0,1,2))))
    # model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))
    model.add(Activation('relu')) # "elu" used in the original paper, relu was used as a variation
    model.add(MaxPooling2D(pool_size = (1,3), strides = (1,3)))
    model.add(Dropout(0.5)) 

    model.add(Conv2D(100, (10), padding = "same", data_format = 'channels_last', kernel_constraint = max_norm(2., axis=(0,1,2))))
    # model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (1,3), strides = (1,3)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(200,  (10), padding = "same", data_format = 'channels_last', kernel_constraint = max_norm(2., axis=(0,1,2))))
    # model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (1,3), strides = (1,3)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2, kernel_constraint = max_norm(0.5))) #integer = dimensionality of the output space
    model.add(Activation('softmax'))
    
    # adam_optimiser = keras.optimizers.Adam(lr=0.0001,  decay=0.1)
    adam_optimiser = Adam(lr=0.001)

    model.compile(loss = 'sparse_categorical_crossentropy',
    # model.compile(loss = 'sparse_categorical_crossentropy',
                    optimizer = adam_optimiser, 
                    metrics = ['accuracy'])
    print(model.summary())
    return model #(inputs = keras_tensor, outputs = 'softmax') 

''' 
Early stopping is the strategy to avoid overfitting:
Chosen "monitor" of performance = val_loss
Model will stop training when val_loss increases, even if there are still training epochs
TODO  value for min_delta (i.e. minimum change needs to be determined)
'''
# log_dir = r'C:\Users\Siobhan\Desktop\SET\CSV\log_dir'
def train_model(model, X0, Y0, X1, Y1):

    earlystopping = EarlyStopping(monitor = 'val_loss', 
                        min_delta = 0.001,
                        patience = 10,
                        verbose = 1,
                        mode = 'auto'
                    )
    
    accuracy = model.fit(X0, Y0,
                # epochs = 1,
                epochs = 50,
                validation_data = (X1, Y1), 
                batch_size = None,
                callbacks = [earlystopping],
                shuffle = True
            )
    print(accuracy)
    return accuracy 
def save_model():    
    # Save architecture
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # Save weights
    model.save_weights("weights.h5")
    print("Model saved to disk")

def test_model(test_input, test_label):
    #Load model
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    # Load weights
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights.h5")
    print("Model and weights loaded")
    #Re-compile model and use test data
    adam_optimiser = Adam(lr=0.001)
    loaded_model.compile(loss = 'sparse_categorical_crossentropy',
                    optimizer = adam_optimiser, 
                    metrics = ['accuracy'])
    score = loaded_model.evaluate(test_input, test_label, verbose = 0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

def plot_LR_dist(data_dict):
    # -1: left
    # 1: right
    # totals = {-1: 0, 1: 0}
    all_responses = []

    for person in data_dict:
        for response in data_dict[person]["responses"]:
            # totals[response[0]] += 1
            all_responses.append(response[0])

    plt.figure()

    plot_data = all_responses
    # plot_data = [totals[-1], totals[1]]
    plt.hist(plot_data)
    plt.title("full data")

    plt.savefig("full data")
    # plt.show()

def plot_LR_arr(array, title):
    reshaped_array = np.reshape(array, (-1,))
    plt.figure()

    plt.hist(reshaped_array)
    plt.title(title)

    # plt.show()
    plt.savefig(title)

if __name__ == "__main__":

    print("PREPARE DATASET V1")

    dataset_25pct = PrepareDataset(
        # raw_csv_data_dir="E:\\SET\\CSV\\onepointfive",
        raw_csv_data_dir="C:\\Users\\Siobhan\\Desktop\\SET\\CSV",
        train_test_split_method="grouped",
        # train_test_split_method="separated",
        train_test_split_pct=0.7,
        prediction_time_step= 0.25)

    # force_data_rescan = False
    force_data_rescan = True
    if force_data_rescan:
        print("Rescanning data...")
        dataset_25pct.read_data_from_raw_csv_data_dir()
        dataset_25pct.split_raw_data_into_frames()
        dataset_25pct.convert_to_ml_dataset()
    else:
        print("Loading cached data...")
        try:
            file_ref = np.load("cached_ml_dataset.npz", allow_pickle=True)
            dataset_25pct.ml_data = file_ref["arr_0"].item()
            file_ref.close()
        except:
            raise

    plot_LR_dist(dataset_25pct.ml_data)

    # X0, Y0, X1, Y1, test_input, test_label = dataset_25pct.split_ml_dataset()
    X0, Y0, test_input, test_label, X1, Y1 = dataset_25pct.split_ml_dataset()

    plot_LR_arr(Y0, "training")
    plot_LR_arr(Y1, "validation")
    plot_LR_arr(test_label, "test")
    # exit()

    # shuffle data to mitigate catastrophic forget
    indices = np.arange(X0.shape[0])
    np.random.shuffle(indices)
    X0 = X0[indices]
    Y0 = Y0[indices]

    print(">>> SEPERATED")
    print("X0", X0.shape, "Y0", Y0.shape, "X1", X1.shape, "Y1", Y1.shape)

    # dataset_25pct = PrepareDataset(
    #     raw_csv_data_dir="C:\\Users\\Siobhan\\Desktop\\SET\\CSV_SMALL",
    #     train_test_split_method="grouped",
    #     train_test_split_pct=0.67,
    #     prediction_time_step=0.50)

    # dataset_25pct.read_data_from_raw_csv_data_dir()
    # dataset_25pct.split_raw_data_into_frames()
    # dataset_25pct.convert_to_ml_dataset()
    # # dataset_25pct.print_shape_of_dataset()

    # X0, Y0, X1, Y1 = dataset_25pct.split_ml_dataset()

    # print(">>> GROUPED")
    # print("X0", X0.shape, "Y0", Y0.shape, "X1", X1.shape, "Y1", Y1.shape)
    '''
    Note about changing the range of the labels:
    - Change range of labels to be [0, 1] because keras' loss
        functions expect it to be so (I think, it gave strange behaviou with [-1, 1]).
    - Also flatten label array (to 1D) - just because I (Elan) wanted to see if that was
        also causing errors, doesn't seem to matter if it is flat or 2D.
    '''
    # print(Y0)
    # print(Y1)
    Y0 = Y0.reshape((-1,)) / 2 + 0.5
    Y1 = Y1.reshape((-1,)) / 2 + 0.5
    test_label = test_label.reshape((-1,)) / 2 + 0.5

    # print(Y0)
    # print(Y1)

    # fix shape of X0 (add "colour" channel to input [to mimic that of an image])
    X0 = X0[:, :, :, np.newaxis]
    X0_shape = X0.shape[1:] # remove input dimension of shape - keras does not want this defined
    model = create_model(X0_shape)
    # shows that model works (we pushed the training data through it)
    print(model.predict(X0))

    # fix shape of X1 (like we did with X0 - add "colour" channel)
    X1 = X1[:, :, :, np.newaxis]
    test_input = test_input[:, :, :, np.newaxis]
    train_model(model, X0, Y0, X1, Y1)
# Save model to .json [model.json] file and weights to a .h5 file [weights.json]
# Saved in code directory
    save_model()
# Only call this function when ready to test - final round. Model can't have seen the data
    # test_model(X0, Y0)
    # test_model(X1, Y1)
    test_model(test_input, test_label)


