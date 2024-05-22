import time

import chess.pgn
import tensorflow as tf
import multiprocessing as mp
import numpy as np
import copy
import sys
start_moves = 0


class SimplifiedBoard():
    def __init__(self):
        self.board_state = np.zeros(shape = (8, 8))
        self.reset_board()
        self.conversion_table = {".": 0, "p": 2, "n": 4, "b": 6, "r": 8, "q": 10, "k": 12, "P": 1, "N": 3, "B": 5, "R": 7, "Q": 9, "K": 11}


    def board_to_list(self, new_board):
        rows = str(new_board).split("\n")
        return [x.split() for x in rows]

    def reset_board(self):
        self.board_state = np.array([[8, 4, 6, 10, 12, 6, 4, 8],
                                     [2, 2, 2, 2, 2, 2, 2, 2],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1],
                                     [7, 3, 5, 9, 11, 5, 3, 7]])

    def set_board_state_from_chess_game(self, game, return_intermediates = False):
        self.reset_board()
        states = []
        while True:
            game = game.next()
            board_state = self.board_to_list(game.board())

            if game.is_end():
                break

            li = []
            for row_index in range(8):
                row = []
                for col_index in range(8):
                    row.append(self.conversion_table[board_state[row_index][col_index]])
                li.append(row)
            self.board_state = np.array(copy.deepcopy(li))
            states.append(self.get_bitboard())
        if return_intermediates:
            return states

    def set_board_state_from_list(self, li):
        for x in range(8):
            for y in range(8):
                self.board_state[x, y] = li[y][x]

    def get_bitboard(self):
        bitboards = np.zeros((12, 8, 8))
        for x in range(8):
            for y in range(8):
                if self.board_state[x, y] != 0:
                    bitboards[self.board_state[x, y] - 1, x, y] = 1
        return bitboards

def build_eval_model():
    class Evaluator(tf.keras.Model):
        def __init__(self):
            super(Evaluator, self).__init__(name = "Evaluator")
            self.normalization = tf.keras.layers.LayerNormalization()
            self.flatten = tf.keras.layers.Flatten()
            self.dense_1 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_1")
            self.dense_2 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_1")
            self.dense_3 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_1")
            self.dense_4 = tf.keras.layers.Dense(1024, activation = "relu", name = "dense_1")
            self.dense_5 = tf.keras.layers.Dense(3, activation = "softmax", name = "dense_1")

        def call(self, x):
            x = self.flatten(x)
            #x = self.normalization(x)
            x = self.dense_1(x)
            x = self.dense_2(x)
            x = self.dense_3(x)
            x = self.dense_4(x)
            x = self.dense_5(x)
            return x

    return Evaluator()


def get_board_states_from_game(file):
    winner = file.headers["Result"]
    if winner == "0-1":
        winner = [0, 1, 0]
    elif winner == "1-0":
        winner = [1, 0, 0]
    else:
        winner = [0, 0, 1]
    board_states = board.set_board_state_from_chess_game(file, return_intermediates = True)[start_moves:]
    return board_states, [winner] * len(board_states)


board = SimplifiedBoard()
if __name__ == "__main__":
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(2000)
    pgn = open("lichess_db_standard_rated_2024-01.pgn", encoding = "utf-8")
    game_files = []

    num_games = 10_000
    for i in range(num_games):
        game = chess.pgn.read_game(pgn)
        game_files.append(game)

    print("loaded")
    # print(get_board_states_from_game(game_files[0]))

    start_time = time.perf_counter()
    with mp.Pool(processes = 16) as pool:
        processed_games = pool.map(get_board_states_from_game, game_files)
    #processed_games = [get_board_states_from_game(q) for q in game_files]
    end_time = time.perf_counter()
    print(end_time - start_time)

    x_train_data = []
    y_train_data = []
    for i, item in enumerate(processed_games):

        for j in range(len(item[0])):
            x_train_data.append(item[0][j])
            y_train_data.append(item[1][j])

    x_train_data = np.array(x_train_data)
    y_train_data = np.array(y_train_data)

    print(x_train_data.shape, y_train_data.shape)

    evaluator = build_eval_model()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    evaluator.compile(optimizer = optimizer, loss = loss, metrics = [accuracy])

    evaluator.fit(x = x_train_data, y = y_train_data, epochs = 5)
    evaluator.summary()
    evaluator.save("Evaluator")
