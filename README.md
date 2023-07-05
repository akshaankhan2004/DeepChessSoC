# DeepChess SoC
I try to create an AI Chess playing model based on the paper DeepChess.
Data is prepared using CCRL datasets available on their website.
15 positions are randomly taken from a game(either a win or a loss), and then converted to bitstring format for training.
It contains two neural nets, Pos2Vec and Siamese network, where the first one is trained unsupervised and the weights acquired are then initialised to Siamese network, which is then trained using Supervised Learning.
Dropout regularisation is used to reduce overfitting.
It is still in training process and the provisional code is available.
# Problems
The model seem to give a Test accuracy of 87%, but the training accuracy seems to be 78.86%, considerably low, which means the model is underfitting.
