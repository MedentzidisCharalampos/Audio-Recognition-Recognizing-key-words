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
