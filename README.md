# Audio-Recognition-Recognizing-key-words
A basic speech recognition network that recognizes ten different words: "down", "go", "left", "no", "right", "stop", "up" and "yes".  


# The dataset

The Speech Commands dataset (https://www.tensorflow.org/datasets/catalog/speech_commands): An audio dataset of spoken words designed to help train and evaluate keyword spotting systems. into a numerical tensor.Its primary goal is to provide a way to build and test small models that detect when a single word is spoken, from a set of ten target words, with as few false positives as possible from background noise or unrelated speech.

# Data Preprocessing

The audio file will initially be read as a binary file and will be converted into a numerical tensor, a wav-encoded audio. A WAV file contains timeseries data with a set number of samples per second. A  few audio waveforms with their corresponding labels:

![alt text](https://github.com/MedentzidisCharalampos/Audio-Recognition-Recognizing-key-words/blob/main/audio_waveforms.png)
