# SpokenDigitRecognition

A Project to recognize spoken digits via a convolutionary neural network

Converts labeled audio files to spectrograms via short time fourier transforms, which are then used to train a convolutionary neural network.

Libraries: PyTorch, NumPy, Pandas, Librosa, Matplotlib

Training: Running main.py will initiate a 25 epoch training cycle utilizing 80% of the data, followed by an accuracy test on the remaining 20% of the data. This test has shown that the model reaches well over 90% accuracy with dropout layers used in training and a 0.001 learning rate.
