import numpy as np
import torch
from torch import nn


def initialize_weights(input_shape, num_neurons, value_range=[0, 1]):
    weights = np.random.uniform(value_range[0], value_range[1], (input_shape, num_neurons))
    bias = np.random.uniform(value_range[0], value_range[1], (1, num_neurons))
    return weights, bias


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return 1. * (x > 0)


def tanh(x):
    return np.tanh(x)


def derivative_tanh(x):
    return 1 - np.tanh(x) ** 2


def mse(y, y_predicted):
    return np.mean(np.square(y - y_predicted))


def derivative_mse(y, y_predicted):
    return 2 * (y_predicted - y) / y.size


def log_loss(y, y_predicted):
    return -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))


def derivative_log_loss(y, y_predicted):
    return (y_predicted - y) / (y_predicted * (1 - y_predicted))


def classify(model, x):
    Z1 = np.dot(x, model["W1"]) + model["b1"]
    A1 = relu(Z1)
    Z2 = np.dot(A1, model["W2"]) + model["b2"]
    A2 = relu(Z2)

    return {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }


def update_weights(model, x, y, cache, alpha=0.1):
    dA2 = derivative_mse(y, cache.get("A2"))
    dZ2 = dA2 * derivative_relu(cache.get("Z2"))
    dW2 = np.dot(cache.get("A1").T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, model["W2"].T)
    dZ1 = dA1 * derivative_relu(cache.get("Z1"))
    dW1 = np.dot(x.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    model["W1"] -= alpha * dW1
    model["b1"] -= alpha * db1
    model["W2"] -= alpha * dW2
    model["b2"] -= alpha * db2

    return model


def classify_hexapawn(model, x):
    Z1 = np.dot(x, model["W1"]) + model["b1"]
    A1 = tanh(Z1)
    Z2 = np.dot(A1, model["W2"]) + model["b2"]
    A2 = tanh(Z2)

    return {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }


def update_weights_hexapawn(model, x, y, cache, alpha=0.1):
    dA2 = derivative_mse(y, cache.get("A2"))
    dZ2 = dA2 * derivative_tanh(cache.get("Z2"))
    dW2 = np.dot(cache.get("A1").T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, model["W2"].T)
    dZ1 = dA1 * derivative_tanh(cache.get("Z1"))
    dW1 = np.dot(x.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    model["W1"] -= alpha * dW1
    model["b1"] -= alpha * db1
    model["W2"] -= alpha * dW2
    model["b2"] -= alpha * db2

    return model


def train_nn(X, y, learning_rate=0.1, epochs=1000):
    W1, b1 = initialize_weights(2, 2)
    W2, b2 = initialize_weights(2, 2)
    model_weights = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }

    for epoch in range(epochs):
        zipped_data = list(zip(X, y))
        np.random.shuffle(zipped_data)
        shuffled_X, shuffled_y = zip(*zipped_data)
        cache = classify(model_weights, np.array(shuffled_X))
        if (epoch+1) % 100 == 0:
            loss = mse(np.array(shuffled_y), cache.get('A2'))
            print(f"Epoch: {epoch+1}\t Loss: {loss}")
        model_weights = update_weights(model_weights, np.array(shuffled_X), np.array(shuffled_y), cache, learning_rate)

    y_test = classify(model_weights, np.array(X))
    print("Predictions:")
    print(y_test.get("A2"))


class TorchNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(TorchNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 2)
        nn.init.uniform_(self.fc1.weight)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2, output_shape)
        nn.init.uniform_(self.fc2.weight)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


def train_torch_nn(X, y, learning_rate, epochs):
    model = TorchNN(2, 2)
    print(model)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        zipped_data = list(zip(X, y))
        np.random.shuffle(zipped_data)
        shuffled_X, shuffled_y = zip(*zipped_data)

        optimizer.zero_grad()

        Y_train_predicted = model(torch.tensor(shuffled_X, dtype=torch.float))

        cost = loss(Y_train_predicted, torch.tensor(
            shuffled_y, dtype=torch.float))
        cost.backward()

        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch: {epoch+1}\t Loss: {cost.item()}")

    Y_test_predicted = model(torch.tensor(X, dtype=torch.float))
    Y_test_predicted = 1 * (Y_test_predicted > 0.5)
    print("Predictions:")
    print(Y_test_predicted)


class TorchNN_Hexapawn(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(TorchNN_Hexapawn, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(64, output_shape)
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        return x
    

def train_torch_hexapawn(X, y, epochs, learning_rate, num_tests=10):
    model = TorchNN_Hexapawn(10, 9)
    print(model)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        zipped_data = list(zip(X, y))
        np.random.shuffle(zipped_data)
        shuffled_X, shuffled_y = zip(*zipped_data)

        optimizer.zero_grad()

        Y_train_predicted = model(torch.tensor(shuffled_X, dtype=torch.float))

        cost = loss(Y_train_predicted, torch.tensor(shuffled_y, dtype=torch.float))
        cost.backward()

        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch: {epoch+1}\t Loss: {cost.item()}")

    inputs = []
    actuals = []
    predictions = []
    predictions_processed = []
    for _ in range(num_tests):
        idx = np.random.randint(0, len(X))
        predicted = model(torch.tensor(X[idx], dtype=torch.float))
        print(f"Input: {X[idx]}\n")
        print(f"Output: {y[idx]}\n")
        print(f"Predicted: {predicted}\n")
        inputs.append(X[idx])
        actuals.append(y[idx])
        predictions.append(predicted)
        local_predictions = []
        for val in list(predicted.detach().numpy()):
            if val >= 0.33:
                local_predictions.append(1)
            elif val <= -0.33:
                local_predictions.append(-1)
            else:
                local_predictions.append(0)
        predictions_processed.append(local_predictions)
    
    number_of_incorrect_predictions = 0
    print("Incorrect Predictions:")
    for val_x, val_y, val_p_pred, val_p_pred_proc in zip(inputs, actuals, predictions, predictions_processed):
        if val_y != val_p_pred_proc:
            number_of_incorrect_predictions += 1
            print(f"Input: {val_x}")
            print(f"Actual: {val_y}")
            print(f"Predicted: {val_p_pred}")
            print(f"Predicted Processed: {val_p_pred_proc}\n")
    print(f"Number of incorrect predictions: {number_of_incorrect_predictions}")
    print(f"Accuracy: {calculate_accuracy(actuals, predictions_processed)}")


def calculate_accuracy(y, y_predicted):
    number_of_correct_predictions = 0
    total_predictions = 0
    for val_y, val_p_pred in zip(y, y_predicted):
        total_predictions += 1
        if val_y == val_p_pred:
            number_of_correct_predictions += 1
    return number_of_correct_predictions / total_predictions


if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0, 0], [0, 1], [0, 1], [1, 0]]
    epochs = 1000
    learning_rate = 0.1

    train_nn(X, y, learning_rate, epochs)
    # train_torch_nn(X, y, learning_rate, epochs)
