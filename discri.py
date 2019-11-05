# Imports
from __future__ import print_function, division
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.contrib.keras.api.keras.layers import BatchNormalization, Activation
from tensorflow.contrib.keras.api.keras.layers import LeakyReLU, LSTM, SimpleRNN, GRU, Bidirectional
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.optimizers import Adam, RMSprop
from tensorflow.contrib.keras.api.keras.backend import expand_dims
from tensorflow.contrib.keras.api.keras.layers import UpSampling1D, Conv1D, LocallyConnected1D
from tensorflow.contrib.keras.api.keras.activations import softmax, tanh, relu
from keras.utils import plot_model
import tensorflow as tf
import argparse
import string
import sys
import numpy as np
import csv

# Disable Warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# Constants
save_interval = 100
learning_rate = 0.0002   # 0.0002

# Arguments
input_benign = 'data/Kaggle_good_10T.txt'
input_phishing = 'data/PhishTank_bad_10T.txt'
data_fake_success = ''
url_len = 1000
batch_size = 64
print_size = 500
epochs = 2000
epochs_D = 5000     #初始化训练discriminator，如果训练次数超过该值，而 acc < threshold，就停止
threshold_D = 0.9   #初始化训练discriminator，使之达到 acc >= threshold
ave = 100           #对acc, d_loss求平均的区间长度
noise_shape = 150
generator_layers = "8:8"
discriminator_layers = "8:4:2"
generator_activation = "tanh"
discriminator_activation = "tanh"
dropout_value = 0.5
discriminator_savefile = "/dev/null"
generator_savefile = "/dev/null"
generated_savefile = "output.txt"
evalD_savefile = "evaluation/evaluation_discri_version1.txt"

# Create noise_shape
noise_shape=(noise_shape,)

# Define Alphabet
alphabet = string.ascii_lowercase + string.digits + "/:._-()=;?&%[]+" # MUST BE EVEN
dictionary_size = len(alphabet) + 1
url_shape = (url_len, dictionary_size)


def main():
    # Define Functions
    build_discriminator = build_discriminator_dense

    # Build dictionary
    dictionary = {}
    reverse_dictionary = {}
    for i, c in enumerate(alphabet):
        dictionary[c] = i + 1
        reverse_dictionary[i + 1] = c

    # Build Oprimizer
    optimizer = Adam(learning_rate, 0.5)

    # Build and compile the discriminator
    print("*** BUILDING DISCRIMINATOR ***")
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])

    # Load the dataset
    data_benign = []
    for line in open(input_benign, "r").read().splitlines():
        this_sample = np.zeros(url_shape)
        line = line.lower()
        if len(set(line) - set(alphabet)) == 0 and len(line) < url_len:
            for i, position in enumerate(this_sample):
                this_sample[i][0] = 1.0

            for i, char in enumerate(line):
                this_sample[i][0] = 0.0
                this_sample[i][dictionary[char]] = 1.0
            data_benign.append(this_sample)
        else:
            print("Uncompatible line:", line)
    print("Data_real_good ready. Lines:", len(data_benign))
    No_benign = len(data_benign)
    X_train_benign = np.array(data_benign)
    print("Array X_train_benign Shape:", X_train_benign.shape)

    data_phishing = []
    for line in open(input_phishing, "r").read().splitlines():
        this_sample = np.zeros(url_shape)
        line = line.lower()
        if len(set(line) - set(alphabet)) == 0 and len(line) < url_len:
            for i, position in enumerate(this_sample):
                this_sample[i][0] = 1.0

            for i, char in enumerate(line):
                this_sample[i][0] = 0.0
                this_sample[i][dictionary[char]] = 1.0
            data_phishing.append(this_sample)
        else:
            print("Uncompatible line:", line)
    print("Data_phishing ready. Lines:", len(data_phishing))
    No_phishing = len(data_phishing)
    X_train_phishing = np.array(data_phishing)
    print("Array X_train_phishing Shape:", X_train_phishing.shape)
    half_batch = int(batch_size / 2)


    # ----------------------------------
    #  Train Discriminator -- Version_1
    # ----------------------------------
    RES = []
    sum_acc = 0
    sum_dloss = 0
    ini_acc_D = 0
    for epoch in range(epochs_D):
        # Select a random half batch of data
        idx = np.random.randint(0, X_train_benign.shape[0], half_batch)
        samples_benign = X_train_benign[idx]
        idx = np.random.randint(0, X_train_phishing.shape[0], half_batch)
        samples_phishing = X_train_phishing[idx]

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(samples_benign, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(samples_phishing, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        ini_acc_D = d_loss[1]
        sum_acc = sum_acc + ini_acc_D
        sum_dloss = sum_dloss + d_loss[0]

        # Plot the progress
        print("%d [D loss: %0.3f, acc.: %0.3f%%]" % (epoch, d_loss[0], 100 * ini_acc_D))

        if epoch % ave == 0:
            record = {}
            record['accuracy'] = sum_acc / ave
            record['d_loss'] = sum_dloss / ave
            RES.append(record)
            sum_acc = 0
            sum_dloss = 0
            print("The average is : %d [D loss: %0.3f, acc.: %0.3f%%]" % (epoch, record['d_loss'], 100 * record['accuracy']))

        if ini_acc_D >= threshold_D:
            break
    # save the RES to csv
    csv_columns = ['accuracy', 'd_loss']
    try:
        with open(evalD_savefile, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in RES:
                writer.writerow(data)
    except IOError:
        print("I/O error")

def build_discriminator_dense():
    model = Sequential()
    model.add(Flatten(input_shape=url_shape))   # add input layer

    # Add arbitrary layers
    for size in discriminator_layers.split(":"):
        size = int(size)
        model.add(Dense(size, activation=discriminator_activation))
        model.add(Dropout(dropout_value))

    # Add the final layer, with a single output
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # Build the model
    gen = Input(shape=url_shape)
    validity = model(gen)
    return Model(gen, validity)


# Main
if __name__ == '__main__':
    main()
