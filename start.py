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

# Disable Warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# Constants
save_interval = 100
learning_rate = 0.001   # 0.0002

# Arguments
input_data = 'Kaggle_good.txt'
url_len = 1000
batch_size = 128
print_size = 500
epochs = 2000
noise_shape = 150
generator_layers = "8:8"
discriminator_layers = "8:4:2"
generator_activation = "tanh"
discriminator_activation = "tanh"
dropout_value = 0.5
discriminator_savefile = "/dev/null"
generator_savefile = "/dev/null"
generated_savefile = "output.txt"


# Create noise_shape
noise_shape=(noise_shape,)

# Define Alphabet
alphabet = string.ascii_lowercase + string.digits + "/:._-()=;?&%[]+" # MUST BE EVEN
dictionary_size = len(alphabet) + 1
url_shape = (url_len, dictionary_size)


def main():
    # Define Functions
    build_generator = build_generator_dense
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

    # Build and compile the generator
    print("*** BUILDING GENERATOR ***")
    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # The generator takes noise as input and generated samples
    # Input() 规定了输入层 input layer 以啥作为输入
    z = Input(shape=noise_shape)
    gen = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The valid takes generated samples as input and determines validity
    valid = discriminator(gen)

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates samples => determines validity
    combined = Model(inputs=[z], outputs=[valid])
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # # plot the combined structure of the two models
    # plot_model(combined, to_file='combined structure.png', show_shapes=True)

    # Load the dataset
    data = []
    for line in open(input_data, "r").read().splitlines():
        this_sample = np.zeros(url_shape)

        line = line.lower()
        # print('打印是否compatible URL，len(set(line) - set(alphabet)) = ', len(set(line) - set(alphabet)))
        # print('len(line) = ', len(line))

        if len(set(line) - set(alphabet)) == 0 and len(line) < url_len:
            for i, position in enumerate(this_sample):
                this_sample[i][0] = 1.0

            for i, char in enumerate(line):
                this_sample[i][0] = 0.0
                this_sample[i][dictionary[char]] = 1.0
            data.append(this_sample)
        else:
            print("Uncompatible line:", line)

    print("Data ready. Lines:", len(data))
    X_train = np.array(data)
    print("Array Shape:", X_train.shape) #Array Shape: (26096, 1000, 48)
    half_batch = int(batch_size / 2)

    # Start Training
    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of data
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        samples = X_train[idx]
        noise_batch_shape = (half_batch,) + noise_shape
        noise = np.random.normal(0, 1, noise_batch_shape)

        # Generate a half batch of new data
        gens = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(samples, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gens, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise_batch_shape = (batch_size,) + noise_shape
        noise = np.random.normal(0, 1, noise_batch_shape)

        # The generator wants the discriminator to label the generated samples as valid (ones)
        valid_y = np.array([1] * batch_size)

        # Train the generator
        g_loss = combined.train_on_batch(noise, valid_y)

        # Plot the progress
        print("%d [D loss: %0.3f, acc.: %0.3f%%] [G loss: %0.3f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # If at save interval, print some examples
        if epoch % save_interval == 0:
            generated_samples = []
            r, c = 5, 5
            noise_batch_shape = (print_size,) + noise_shape
            noise = np.random.normal(0, 1, noise_batch_shape)
            gens = generator.predict(noise)

            for url in gens:
                this_url_gen = ""
                for position in url:
                    this_index = np.argmax(position)
                    if this_index != 0:
                        this_url_gen += reverse_dictionary[this_index]

                print(this_url_gen)
                generated_samples.append(this_url_gen)

    # Save networks
    discriminator.save(discriminator_savefile)
    generator.save(generator_savefile)

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
    print('np.prod(url_shape) : ', np.prod(url_shape))
    model.add(Dropout(dropout_value))
    model.add(Reshape(url_shape))
    print('Reshape(url_shape) : ', Reshape(url_shape))
    model.summary()

    # Build the model
    noise = Input(shape=noise_shape)
    gen = model(noise)

    return Model(noise, gen)


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


