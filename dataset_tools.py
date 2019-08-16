import os
import re
import ffmpeg
from pydub import AudioSegment
import librosa as lr
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def convert_au_to_wav(data_dir, genres):
    """ Converts all .au audio files to .wav """
    for genre in genres:
        current_genre_path = os.path.join(data_dir, genre)

        for root, dirs, files in os.walk(current_genre_path):
            for file in files:

                base_name = file.split('.au')[0]
                (
                    ffmpeg
                    .input(f"{os.path.join(current_genre_path, file)}")
                    .output(f"{os.path.join(current_genre_path, base_name)}.wav")
                    .run()
                )


def slice_audio(audio_file, end=3):
    """ Slices a single audio file into 3 second chunks """
    start = 0
    end *= 1000
    audio = AudioSegment.from_wav(audio_file)
    slices = []

    for i in range(10):
        audio_slice = audio[start:end]
        slices.append(audio_slice)
        start += (1000 * end)
        end += (1000 * end)

    return slices


def slice_dataset(data_dir, genres):
    """ Iterates through entire dataset and converts every audio file to 3 second slices """
    for genre in genres:
        current_index = 0
        current_genre_path = os.path.join(data_dir, genre)

        for root, dirs, files in os.walk(current_genre_path):
            for file in files:
                song_genre = file.split('.')[0]
                audio_file = os.path.join(current_genre_path, file)
                audio_slices = slice_audio(audio_file)

                for audio in audio_slices:
                    os.chdir(current_genre_path)
                    audio.export(f"{genre}.{current_index}.wav", format="wav")
                    current_index += 1


def clean_up_files(data_dir, genre):
    """ Iterates through dataset and removes original whole audio files """
    regexp = re.compile(r"\d\d\d\d\d")
    for genre in genres:
        current_genre_path = os.path.join(data_dir, genre)

        for root, dirs, files in os.walk(current_genre_path):
            for file in files:
                if regexp.search(file):
                    os.remove(os.path.join(root, file))


def build_training_data(data_dir, genres, width, height):
    """ builds training data from audio slices and return an array of image data and label """
    training_data = []

    for genre in genres:
        current_genre_path = os.path.join(data_dir, genre)

        for root, dirs, files in os.walk(current_genre_path):
            for file in files:
                song_genre = file.split('.')[0]
                class_num = genres.index(song_genre)

                song_path = os.path.join(root, file)
                y, sr = lr.load(song_path, sr=22050)

                spectrogram = lr.feature.melspectrogram(y=y, sr=sr)
                img = cv2.resize(spectrogram, (width, height))
                training_data.append([img, song_genre])

    return training_data


def preprocess_data(X, y):
    """ Reshapes arrays and converts labels to one-hot arrays and splits to train and test """
    X = np.array(X).reshape(-1, width, height, 1)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y)

    print('Done...')
    X = X / 255.0  # normalize pixel values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test
