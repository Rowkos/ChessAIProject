import chess.pgn
import tensorflow as tf
import numpy as np
import pickle

def build_eval_model():
    class Evaluator(tf.keras.Model):
        def __init__(self):
            super(Evaluator, self).__init__(name = "Evaluator")
            self.dense_1 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_1")
            self.dense_2 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_2")
            self.dense_3 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_3")
            self.dense_4 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_4")
            self.dense_5 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_5")
            self.dense_6 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_6")
            self.dense_7 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_7")
            self.dense_8 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_8")
            self.dense_9 = tf.keras.layers.Dense(768, activation = "sigmoid", name = "dense_9")

        def call(self, x):
            x = self.dense_1(x)
            x = self.dense_2(x)
            x = self.dense_3(x)
            x = self.dense_4(x)
            x = self.dense_5(x)
            x = self.dense_6(x)
            x = self.dense_7(x)
            x = self.dense_8(x)
            x = self.dense_9(x)
            return x

    return Evaluator()


def board_to_list(board):
    rows = str(board).split("\n")
    return [x.split() for x in rows]


def get_bitboard(state):
    bitboards = np.zeros((12, 8, 8))
    for x in range(8):
        for y in range(8):
            if state[x, y] != 0:
                bitboards[int(state[x, y]) - 1, x, y] = 1
    return bitboards

def board_to_ints(samle_board):
    state = board_to_list(samle_board)
    for row in range(len(state)):
        for col in range(len(state[row])):
            state[row][col] = pieces_to_numbers[state[row][col]]
    state = get_bitboard(np.array(state))
    state = state.flatten()
    return state

pieces_to_numbers = {".": 0, "P": 1, "p": 2, "N": 3, "n": 4, "B": 5, "b": 6, "R": 7, "r": 8, "Q": 9, "q": 10, "K": 11,
                     "k": 12}

with open("GameData", "rb") as file:
    game_files = pickle.load(file)

x_train = []
y_train = []
for i in range(len(game_files)):
    game = game_files[i]
    while True:
        game = game.next()
        if game.is_end():
            break

        board = game.board()
        x_train.append(board_to_ints(board))
        y_train.append(game.eval())

print(len(x_train), len(y_train))

model = build_eval_model()
loss = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(1e-4)

model.compile(optimizer = optimizer, loss = loss)
model.fit(x_train, y_train, epochs = 5, validation_split = 0.1)