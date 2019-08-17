import os
import pickle
import argparse as ag
import warnings
from dataset_tools import convert_au_to_wav, build_training_data, slice_audio
from dataset_tools import slice_dataset, clean_up_files, preprocess_data
from model import create_model
import librosa as lr
import cv2
import numpy as np

# disable warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# list of genres in dataset and path to dataset
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

classifications = {k: v for k, v in enumerate(genres)}

# constants
WIDTH = 128
HEIGHT = 128

def main(args):
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

            # build data and labels from spectrograms
            X, y = build_training_data(data_dir, genres, WIDTH, HEIGHT)

        # prepreprocess data and split into test and training sets
        X_train, X_test, y_train, y_test = preprocess_data(X, y, WIDTH, HEIGHT)
        model = create_model(input_shape=X_train.shape[1:])

        # fit the model to the dataset
        model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=args['epochs'])

        # evaluate model against labels
        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0) 

        print(f"Train Accuracy: {round(train_acc*100, 2)}%\nTest Accuracy: {round(test_acc*100, 2)}%")

    elif args['type'] == 'test':

        # create model and load weights
        model = create_model(input_shape=[128, 128, 1])
        model.load_weights('cnn_weights.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics = ['accuracy'])

        # get slices from song
        song = args['song']
        slices = slice_audio(song)

        # write slices to local directory
        for idx, audio in enumerate(slices):
            audio.export(f"song{str(idx)}.wav", format="wav")

        # convert slices to spectrogram and convert to respective array
        song_data = []
        for i in range(10):
            audio, sr = lr.load(f'song{str(idx)}.wav', sr=22050)
            spectrogram = lr.feature.melspectrogram(y=audio, sr=sr)
            img = cv2.resize(spectrogram, (WIDTH, HEIGHT))
            X = np.array(img).reshape(-1, WIDTH, HEIGHT, 1)
            X /= 255
            song_data.append(X)

        # predict spectrograms passed through network
        predictions = model.predict_classes(song_data)
        print(classifications[predictions[0]])



if __name__ == '__main__':
    # parse command line arguments
    parser = ag.ArgumentParser("Music Genre Classification on the GTZAN Dataset")

    # required arguments
    parser.add_argument('-t', '--type', help='Execute train or test mode',
                        type=str, required=True)
    # optional arguments
    parser.add_argument('-d', '--directory', help='Path to GTZAN dataset',
                        type=str, required=False)
    parser.add_argument('-s', '--song', help='Song to classify (Current Directory Only)',
                        type=str, required=False)
    parser.add_argument('-e', '--epochs', help='Number of epochs to train',
                        type=int, required=False)

    # parse arguments and cast them into dictionary
    args = parser.parse_args()
    args = vars(args)

    main(args)

