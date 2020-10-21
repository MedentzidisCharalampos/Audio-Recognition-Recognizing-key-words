# Audio-Recognition-Recognizing-key-words
A basic speech recognition network that recognizes ten different words: "down", "go", "left", "no", "right", "stop", "up" and "yes".  


# The dataset

The Speech Commands dataset (https://www.tensorflow.org/datasets/catalog/speech_commands): An audio dataset of spoken words designed to help train and evaluate keyword spotting systems. into a numerical tensor.Its primary goal is to provide a way to build and test small models that detect when a single word is spoken, from a set of ten target words, with as few false positives as possible from background noise or unrelated speech.

# Data Preprocessing

The audio file will initially be read as a binary file and will be converted into a numerical tensor, a wav-encoded audio. A WAV file contains timeseries data with a set number of samples per second.   
A few audio waveforms with their corresponding labels:  
![alt text](https://github.com/MedentzidisCharalampos/Audio-Recognition-Recognizing-key-words/blob/main/audio_waveforms.png)

# Spectrogram
We convert the waveform into a spectrogram, which shows frequency changes over time and can be represented as a 2D image. This can be done by applying short time fourier transform (STFT) to convert the audio into the time-frequency domain.

A fourier transform (tf.signal.fft) converts a signal to its component frequencies, but looses all time information. The STFT (tf.signal.stft) splits the signal into windows of time and runs a fourier transform on each window, preserving some time information, and returning a 2D tensor that you can run standard convolutions on.

STFT produces an array of complex numbers representing magnitude and phase. However,we only need the magnitude.


We also want the waveforms to have the same length so that when you convert it to a spectrogram image, the results will have similar dimensions. This can be done by simply zero padding the audio clips that are shorter than one second.

An example from an audio file that has has the word "right". The waveform and the spectrogram is shown below:
![alt text](https://github.com/MedentzidisCharalampos/Audio-Recognition-Recognizing-key-words/blob/main/wav_spec.png)

The spectrogram " for different samples of the dataset:
https://github.com/MedentzidisCharalampos/Audio-Recognition-Recognizing-key-words/blob/main/different_samples_spectogram.png

# Build and Train the Model

For the model, we use a simple convolutional neural network (CNN), since we have transformed the audio files into spectrogram images. The model also has the following additional preprocessing layers:

A Resizing layer to downsample the input to enable the model to train faster.
A Normalization layer to normalize each pixel in the image based on its mean and standard deviation.
For the Normalization layer, its adapt method would first need to be called on the training data in order to compute aggregate statistics (i.e. mean and standard deviation).

The summary of the model is shown below:

nput shape: (124, 129, 1)
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

The model is compiled with SparseCategoricalCrossentropy as Loss function and Adam as optimizer.  
The model is trained for 10 epochs with early stopping to avoid overfitting.  
After the training we have the results `loss: 0.4150 - accuracy: 0.8550 - val_loss: 0.5001 - val_accuracy: 0.8388`
