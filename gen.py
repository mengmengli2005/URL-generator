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

# Constants
save_interval = 100
learning_rate=0.0002

# Arguments
input_data = 'Kaggle_good.txt'
url_len = 50
batch_size = 128
print_size = 500
epochs = 2000
noise_shape = 150
generator_layers = "8:8"
generator_activation = "tanh"
dropout_value = 0.5
generated_savefile = "output_version1.txt"

# Create noise_shape
noise_shape=(noise_shape,)

# Define Alphabet
alphabet = string.ascii_lowercase + string.digits + "/:._-()=;?&%[]" # MUST BE EVEN
dictionary_size = len(alphabet) + 1
url_shape = (url_len, dictionary_size)


def main():
    # build generator
    build_generator = build_generator_dense

    # build dictionary
    dictionary = {}
    reverse_dictionary = {}
    for i, c in enumerate(alphabet):
        dictionary[c] = i + 1
        reverse_dictionary[i + 1] = c

    # Build Oprimizer
    optimizer = Adam(learning_rate, 0.5)

    # Build and compile the generator
    print("*** BUILDING GENERATOR ***")
    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # The generator takes noise as input and generated samples
    # Input() 规定了输入层 input layer 以啥作为输入
    z = Input(shape=noise_shape)
    gen = generator(z)

    #-------------------------------------------------------------------

    # # Load the dataset
    # data = []
    # for line in open(input_data, "r").read().splitlines():
    #     this_sample = np.zeros(url_shape)
    #
    #     line = line.lower()
    #     print('打印是否compatible URL，len(set(line) - set(alphabet)) = ', len(set(line) - set(alphabet)))
    #     print('len(line) = ', len(line))
    #
    #     if len(set(line) - set(alphabet)) == 0 and len(line) < url_len:
    #         for i, position in enumerate(this_sample):
    #             this_sample[i][0] = 1.0
    #
    #         for i, char in enumerate(line):
    #             this_sample[i][0] = 0.0
    #             this_sample[i][dictionary[char]] = 1.0
    #         data.append(this_sample)
    #     else:
    #         print("Uncompatible line:", line)
    #
    # print("Data ready. Lines:", len(data))
    # X_train = np.array(data)
    # print("Array Shape:", X_train.shape)
    # half_batch = int(batch_size / 2)

    noise_batch_shape = (batch_size,) + noise_shape
    print('noise_batch_shape : ', noise_batch_shape)
    noise = np.random.normal(0, 1, noise_batch_shape)
    print('noise : ', noise)
    # Generate a batch of new data
    gens = generator.predict(noise)

    generated_samples = []
    for url in gens:
        this_url_gen = ""
        for position in url:
            this_index = np.argmax(position)
            if this_index != 0:
                this_url_gen += reverse_dictionary[this_index]

        print(this_url_gen)
        generated_samples.append(this_url_gen)

    # Save Samples
    fo = open(generated_savefile, "w")
    for url in generated_samples:
        print(url, file=fo)
    fo.close()


def build_generator_dense():
    model = Sequential()

    # Add arbitrary layers
    first = True
    for size in generator_layers.split(":"):
        size = int(size)
        if first:
            model.add(Dense(size, input_shape=noise_shape, activation=generator_activation))
        else:
            model.add(Dense(size, activation=generator_activation))

        model.add(Dropout(dropout_value))
        first = False

    # Add the final layer
    model.add(Dense(np.prod(url_shape), activation="tanh"))
    model.add(Dropout(dropout_value))
    model.add(Reshape(url_shape))
    model.summary()

    # Build the model
    noise = Input(shape=noise_shape)
    gen = model(noise)

    return Model(noise, gen)

# Main
if __name__ == '__main__':
    main()