import os
import pickle
import argparse as ag
import datetime
import random
import warnings
from dataset_tools import convert_au_to_wav, build_training_data
from dataset_tools import slice_dataset, clean_up_files
from model import create_model


# disable warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# list of genres in dataset and path to dataset
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# constants
WIDHTH = 128
HEIGHT = 128

def main():
    data_dir = args['directory']
    modes = ['train', 'test']

    # raise error if invalid argument is passed
    if args['type'] not in modes:
        raise ValueError("Invalid argument was passed, type \"--help\" for options.")

    if args['type'] == 'train':

        if not args['epochs']:
            raise ValueError("Number of epochs was not specified.")

        # if the training data pickle files exists then load them
        if os.path.exists('x_data.pkl') and os.path.exists('y_data.pkl'):
            with open('x_data.pkl', 'rb') as file:
                X = pickle.load(file)

            with open('y_data.pkl', 'rb') as file:
                y = pickle.load(file)

        else:
            # convert all audio files from .au format to .wav
            convert_au_to_wav(data_dir, genres)

            # slice every song in the data from 30 sec chunks to 3 seconds
            slice_dataset(data_dir, genres)

            # remove all extraneous files
            clean_up_files(data_dir, genres)

            training_data = build_training_data(data_dir, genres, WIDTH, HEIGHT)

            X = [img for img, _ in training_data]
            y = [label for _, label in training_data]

        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        model = create_model(input_shape=X.shape[1:])

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics = ['accuracy'])

        # fit the model to the dataset
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=args['epochs'])

        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0) 



if __name__ == '__main__':
    # parse command line arguments
    parser = ag.ArgumentParser("Music Genre Classification on the GTZAN Dataset")

    # required arguments
    parser.add_argument('-t', '--type', help='Execute train or test mode', 
                        type=str, required=True)
    # optional arguments
    parser.add_argument('-d', '--directory', help='Path to GTZAN dataset',
                        type=str, required=False)
    parser.add_argument('-s', '--song', help='Song to classify (full path)',
                        type=str, required=False)
    parser.add_argument('-e', '--epochs', help='Number of epochs to train',
                        type=int, required=False)

    # parse arguments and cast them into dictionary
    args = parser.parse_args()
    args = vars(args)

    main(args)

