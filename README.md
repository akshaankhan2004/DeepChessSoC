To run the files, download whole "FinalEngine" folder to desktop and run the 'BestMove.ipynb' file. You can input desired position to get the best move.
# DeepCarlsen SoC
This is my repository for Summer of Code 2023 project under Web and Coding Club of IIT Bombay, under the mentorship of my SoC mentors Navyansh Mahla and Atishay Jain.

I try to create an AI Chess playing model based on the paper DeepChess.
Data is prepared using CCRL datasets available on their website.
15 positions are randomly taken from a game(either a win or a loss), and then converted to bitstring format for training.
It contains two neural nets, Pos2Vec and Siamese network, where the first one is trained unsupervised and the weights acquired are then initialised to Siamese network, which is then trained using Supervised Learning.
Pos2Vec is trained for 100 epochs for each autoencoder while Siamese network is trained for 500 epochs.
Dropout regularisation is used to reduce overfitting.
# Problems
The model seem to give a Train accuracy of 87%, but the testing accuracy seems to be 78.86%, considerably low, which means the model is overfitting.

Preferable depth to play would be 3 as it makes kind of good moves with it but it takes a lot of time, around 3-5 minutes for one move (on GPU), depending upon the complexity of the position, meanwhile increasing more depth would take a lot of time, and with depth of 1 or 2, it plays recklessly. Looking for the ways to decrease this time complexity.

I used an online ELO predictor to check how well it plays, which checked it by giving 10 random positions, it predicted an ELO of 1350 for puzzles when played with the depth of 3. (ELO might not be an accurate one and can be rounded off for reference)
# References and Resources
TensorFlow tutorial- https://www.tensorflow.org/tutorials


DeepChess Research Paper- https://arxiv.org/pdf/1711.09667.pdf


Mentor GitHub Repo for Resources- https://github.com/navyanshmahla/deep-carlsen-SoC23
