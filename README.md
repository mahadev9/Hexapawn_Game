# Hexapawn Game
Implemented Minimax Algorithm to build policy table and trained neural network to play hexapawn.

### Requirements
Python 3.10

### Install Dependencies
```console
pip install -r requirements.txt
```

### Run the program
```console
python main.py
```

### Neural Network Architecture
Input shape: 10
Output shape: 9
Number of hidden layers: 1
Number of neurons for each hidden layer: 64
Activation function after each layer: Tanh

Loss function: Mean Squared Error
Learning Rate: 0.1
Epochs: 10000

Note: This project has been implemented using PyTorch as well. This has been done to understand the behaviour of neural network with different hyperparameters. Coincidentally, PyTorch model performs slightly better. To run with PyTorch, follow below instructions:
1. Comment line number 131 and 132 in the file main.py
2. Uncomment line number 133 in the file main.py
3. Use the above run command.

