#Setup

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#Import the Speech Commands dataset

data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

#Check basic statistics about the dataset.

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

# Commands: ['down' 'no' 'go' 'yes' 'stop' 'up' 'right' 'left']

#Extract the audio files into a list and shuffle the list.

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

#Number of total examples: 8000
#Number of examples per label: 1000
#Example file tensor: tf.Tensor(b'data/mini_speech_commands/right/2be08ce4_nohash_0.wav', shape=(), dtype=string)




#Split the files into training, validation and test sets using a 80:10:10 ratio respectively.

train_files = filenames[:6400]
val_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

#Training set size 6400
#Validation set size 800
#Test set size 800

#Reading audio files and their labels

#The audio file will initially be read as a binary file, which you'll want to convert into a numerical tensor.

#To load an audio file, you will use tf.audio.decode_wav, which returns the WAV-encoded audio as a Tensor and the sample rate.

#A WAV file contains timeseries data with a set number of samples per second. Each sample represents the amplitude of the audio signal at that specific time.
# In a 16-bit system, like the files in mini_speech_commands, the values range from -32768 to 32767.
# The sample rate for this dataset is 16kHz.
# Note that tf.audio.decode_wav will normalize the values to the range [-1.0, 1.0].

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

#The label for the wav file is its parent directory.

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)

  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

#Let's define a method that will take in the filename of the wav file and output an audio, label tuple for supervised training.

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

# We now apply process_path to build our training set to extract audio, label pairs and check the results.
# We build the validation and test sets using a similar procedure later on.

AUTOTUNE = tf.data.experimental.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

#Let's examine a few audio waveforms with their corresponding labels.

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
for i, (audio, label) in enumerate(waveform_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  ax.plot(audio.numpy())
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
  label = label.numpy().decode('utf-8')
  ax.set_title(label)

plt.show()

#Spectrogram
#We convert the waveform into a spectrogram, which shows frequency changes over time and can be represented as a 2D image.
# This can be done by applying short time fourier transform (STFT) to convert the audio into the time-frequency domain.

#A fourier transform (tf.signal.fft) converts a signal to its component frequencies, but looses all time information.
# The STFT (tf.signal.stft) splits the signal into windows of time and runs a fourier transform on each window, preserving some time information, and returning a 2D tensor that you can run standard convolutions on.

#STFT produces an array of complex numbers representing magnitude and phase. However, we only need the magnitude , which can be derived by applying tf.abs on the output of tf.signal.stft.

#Choose frame_length and frame_step parameters such that the generated spectrogram "image" is almost square.

#You also want to waveforms to have the same length so that when you convert it to a spectrogram image, the results will have similar dimensions.
# This can be done by simply zero padding the audio clips that are shorter than one second.

def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


#Next, we explore the data.Compare the waveform, the spectrogram and the actual audio of one example from he dataset.

for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)
print('Audio playback')
display.display(display.Audio(waveform, rate=16000))

#Label: right
#Waveform shape: (12800,)
#Spectrogram shape: (124, 129)

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  X = np.arange(16000, step=height + 1)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

#Examine the spectrogram "images" for different samples of the dataset.

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
    ax.set_title(commands[label_id.numpy()])
    ax.axis('off')

plt.show()

# Build and train the model

#Before you build and train our model, you would need apply the same data processing done on the training set with the validation and test sets.

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

#Batch the training and validation sets for model training.

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

#Add dataset cache() and prefetch() operations to reduce read latency while training the model.

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

#For the model, you'll use a simple convolutional neural network (CNN), since you have transformed the audio files into spectrogram images.
# The model also has the following additional preprocessing layers:

#A Resizing layer to downsample the input to enable the model to train faster.
#A Normalization layer to normalize each pixel in the image based on its mean and standard deviation.
#For the Normalization layer, its adapt method would first need to be called on the training data in order to compute aggregate statistics (i.e. mean and standard deviation).

"""
Input shape: (124, 129, 1)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
resizing (Resizing)          (None, 32, 32, 1)         0         
_________________________________________________________________
normalization (Normalization (None, 32, 32, 1)         3         
_________________________________________________________________
conv2d (Conv2D)              (None, 30, 30, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 12544)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               1605760   
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 1032      
=================================================================
Total params: 1,625,611
Trainable params: 1,625,608
Non-trainable params: 3
_________________________________________________________________
"""

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

"""

100/100 [==============================] - 0s 4ms/step - loss: 0.5922 - accuracy: 0.7887 - val_loss: 0.5791 - val_accuracy: 0.8100
Epoch 7/10
100/100 [==============================] - 0s 4ms/step - loss: 0.5295 - accuracy: 0.8128 - val_loss: 0.5632 - val_accuracy: 0.8112
Epoch 8/10
100/100 [==============================] - 0s 4ms/step - loss: 0.4716 - accuracy: 0.8322 - val_loss: 0.5539 - val_accuracy: 0.8325
Epoch 9/10
100/100 [==============================] - 0s 4ms/step - loss: 0.4326 - accuracy: 0.8427 - val_loss: 0.5095 - val_accuracy: 0.8263
Epoch 10/10
100/100 [==============================] - 0s 4ms/step - loss: 0.4150 - accuracy: 0.8550 - val_loss: 0.5001 - val_accuracy: 0.8388
"""

# check the training and validation loss curves to see how our model has improved during training.

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()



for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

#Evaluate test set performance
#Let's run the model on the test set and check performance.

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

#Test set accuracy: 82%

#Display a confusion matrix

#A confusion matrix is helpful to see how well the model did on each of the commands in the test set.

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

#Run inference on an audio file
#Finally, verify the model's prediction output using an input audio file of someone saying "no." How well does our model perform?

sample_file = data_dir/'no/01bb6a2a_nohash_0.wav'

sample_ds = preprocess_dataset([str(sample_file)])

for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  plt.bar(commands, tf.nn.softmax(prediction[0]))
  plt.title(f'Predictions for "{commands[label[0]]}"')
  plt.show()