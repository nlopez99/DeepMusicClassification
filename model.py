import ffmpeg
import librosa as lr
import os
from tqdm import tqdm
import re
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
from pydub import AudioSegment

def create_model(imput_shape):
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(512, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics = ['accuracy'])

    return model


data_dir = '/home/trident/python_projects/ML/datasets/genres/'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
width, height = 128, 128

def convert_au_to_wav(data_dir, genres):
    for genre in tqdm(genres):
        current_genre_path = os.path.join(data_dir, genre)

        for root, dirs, files in os.walk(current_genre_path):
            for file in files:

                base_name = file.split('.au')[0]
                (
                    ffmpeg
                    .input(f"{str(os.path.join(current_genre_path, file))}")
                    .output(f"{str(os.path.join(current_genre_path, base_name))}.wav")
                    .run()
                )

def slice_audio(audio_file, end=3):
    start = 0
    end *= 1000
    audio = AudioSegment.from_wav(audio_file)
    slices = []

    for i in range(10):
        audio_slice = audio[start:end]
        slices.append(audio_slice)
        start += 3000
        end += 3000

    return slices

    for index, audio in enumerate(slices):
        audio.export(f"{index}.wav", format="wav")

def slice_dataset(data_dir, genres):
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
    regexp = re.compile(r"\d\d\d\d\d")
    for genre in genres:
        current_genre_path = os.path.join(data_dir, genre)

        for root, dirs, files in os.walk(current_genre_path):
            for file in files:
                if regexp.search(file):
                    os.remove(os.path.join(root, file))

def build_training_data(data_dir, genres):
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


# print('Building Training Data...')
# training_data = build_training_data(data_dir, genres)


# X = [img for img, _ in training_data]
# y = [label for _, label in training_data]

with open('x_data.pkl', 'rb') as file:
    X = pickle.load(file)

with open('y_data.pkl', 'rb') as file:
    y = pickle.load(file)

X = np.array(X).reshape(-1, width, height, 1)
y = np.array(y)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

print('Done...')
X = X / 255.0 # normalize pixel values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# create log file for tensorboard for model analysis
NAME = "CNN Music Classifications 500 Epochs"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# create model
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, (3,3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics = ['accuracy'])

# fit the model to the dataset
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=500,
          callbacks=[tensorboard])

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0) 

model.save('cnn_weights2.h5')
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#TODO
# get spectrogram dataset X
# build training data from spectrograms
# normalize spectrogram data
# scale spectrogram data
# pass spectrogram data through model
# evaluate model performance

# 61%