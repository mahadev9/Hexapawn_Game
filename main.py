import numpy as np
from Hexapwan import Hexapawn
import NeuralNetwork as nn


INITIAL_STATE = [-1, -1, -1, -1, 0, 0, 0, 1, 1, 1]


def create_statespace():
    game = Hexapawn(INITIAL_STATE)
    space = set()
    state_queue = [game]
    while state_queue:
        s = state_queue.pop()
        if s.is_terminal():
            continue
        acts = s.actions()
        space.add(tuple(s.board))
        for a in acts:
            state_queue.append(s.result(a))
    return space


def create_policy_table(space):
    table = {}
    for state in space:
        move_list = []
        game = Hexapawn(state)
        next_move = minimax_search(game)
        ga = game.copy()
        while not ga.is_terminal():
            move_list.append(next_move)
            next_move = minimax_search(ga)
            ga = ga.result(next_move)
        u = ga.utility()
        table[tuple(game.board)] = (u, move_list)
    return table


def minimax_search(game):
    value, move = max_value(game)
    return move


def max_value(game):
    move = None
    if game.is_terminal():
        return game.utility(), move
    v = -float('inf')
    for act in game.actions():
        new_v, new_act = min_value(game.result(act))
        if new_v > v:
            v = new_v
            move = act
    return v, move


def min_value(game):
    move = None
    if game.is_terminal():
        return game.utility(), move
    v = float('inf')
    for act in game.actions():
        new_v, new_act = max_value(game.result(act))
        if new_v < v:
            v = new_v
            move = act
    return v, move


def create_training_data(spaces):
    Xs, ys = [], []
    for sp in spaces:
        game = Hexapawn(sp)
        act = minimax_search(game)
        next_sp = game.result(act)
        Xs.append(game.board)
        ys.append(next_sp.board[1:])
    return Xs, ys


def train_hexapawn(X, y, epochs, learning_rate):
    W1, b1 = nn.initialize_weights(10, 64, [-1, 1])
    W2, b2 = nn.initialize_weights(64, 9, [-1, 1])
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
        cache = nn.classify_hexapawn(model_weights, np.array(shuffled_X))
        if (epoch+1) % 100 == 0:
            loss = nn.mse(np.array(shuffled_y), cache.get('A2'))
            print(f"Epoch: {epoch+1}\t Loss: {loss}")
        model_weights = nn.update_weights_hexapawn(model_weights, np.array(shuffled_X), np.array(shuffled_y), cache, learning_rate)

    return model_weights


def test_hexapawn(model, X, y, num_tests=10):
    inputs = []
    actuals = []
    predictions = []
    predictions_processed = []
    for _ in range(num_tests):
        idx = np.random.randint(0, len(X))
        predicted = nn.classify_hexapawn(model, np.array(X[idx]))
        local_predictions = []
        for val in list(predicted.get('A2'))[0]:
            if val >= 0.33:
                local_predictions.append(1)
            elif val <= -0.33:
                local_predictions.append(-1)
            else:
                local_predictions.append(0)
        inputs.append(X[idx])
        predictions_processed.append(local_predictions)
        predictions.append(predicted.get('A2'))
        actuals.append(y[idx])
        print(f"Input: {X[idx]}\n")
        print(f"Output: {y[idx]}\n")
        print(f"Predicted: {predicted.get('A2')}\n")
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
    print(f"Accuracy: {nn.calculate_accuracy(actuals, predictions_processed)}")


if __name__ == "__main__":
    spaces = create_statespace()
    policy_table = create_policy_table(spaces)
    X, y = create_training_data(spaces)
    model = train_hexapawn(X, y, 10000, 0.1)
    test_hexapawn(model, X, y, 25)
    # nn.train_torch_hexapawn(X, y, 10000, 0.1)
