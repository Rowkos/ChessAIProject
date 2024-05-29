import pickle
import chess.pgn
with open("/notebooks/GameData02", "rb") as file:
    game_files = pickle.load(file)
print(len(game_files))

import chess.pgn
import tensorflow as tf
import numpy as np
import pickle
import multiprocessing as mp
import gc


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
            self.dense_9 = tf.keras.layers.Dense(1, activation = "sigmoid", name = "dense_9")

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


class custom_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        manager.save()


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


def get_board_states_from_game(game):
    outputs = []
    while True:
        game = game.next()
        if game.is_end():
            break

        board = game.board()
        try:
            outputs.append([board_to_ints(board), game.eval().wdl().white().expectation()])
        except:
            break
    return outputs


def train_data_generator():
    for i in range(len(game_files)):
        if (i % 10) == 0:
            continue
        game = game_files[i]
        while True:
            game = game.next()
            if game.is_end():
                break

            board = game.board()
            try:
                yield board_to_ints(board), game.eval().wdl().white().expectation()
            except:
                break


def val_data_generator():
    for i in range(len(game_files)):
        if (i % 10) != 0:
            continue
        game = game_files[i]
        while True:
            game = game.next()
            if game.is_end():
                break

            board = game.board()
            try:
                yield board_to_ints(board), game.eval().wdl().white().expectation()
            except:
                break

    '''x_train = []
    y_train = []

    with mp.Pool(processes = 16) as pool:
        processed_data = pool.map(get_board_states_from_game, game_files[index * 10_000:(index + 1) * 10_000])

    for i in range(len(processed_data)):
        x_train += [x[0] for x in processed_data[i]]
        y_train += [x[1] for x in processed_data[i]]

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    from sklearn.utils import shuffle
    x_train, y_train = shuffle(x_train, y_train)

    print(x_train[:5], y_train[:5])
    print(len(x_train), len(y_train))
    return x_train, y_train'''


model = build_eval_model()
loss = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(1e-4)
x_train = None
y_train = None

model.compile(optimizer = optimizer, loss = loss)

dataset = tf.data.Dataset.from_generator(train_data_generator,
                                         output_signature = (
                                             tf.TensorSpec(shape = (768), dtype = tf.float32),
                                             tf.TensorSpec(shape = (), dtype = tf.float32))) \
    .batch(32, drop_remainder = True).shuffle(buffer_size = 100_000).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_generator(val_data_generator,
                                             output_signature = (
                                                 tf.TensorSpec(shape = (768), dtype = tf.float32),
                                                 tf.TensorSpec(shape = (), dtype = tf.float32))) \
    .batch(32, drop_remainder = True).prefetch(tf.data.AUTOTUNE)
print(list(dataset.take(1)))

model_path = "ChessAI/PositionEvaluator"
checkpoint = tf.train.Checkpoint(optimizer = optimizer, model = model)
manager = tf.train.CheckpointManager(checkpoint, directory = model_path, max_to_keep = 10)
manager.restore_or_initialize()

cp_callback = custom_callback()
tb_callback = tf.keras.callbacks.TensorBoard(log_dir = "ChessAI/PositionEvaluator01", histogram_freq = 1)
model.fit(dataset, validation_data = val_dataset, epochs = 5, callbacks = [tb_callback, cp_callback])

model.save_weights('ChessAIcheckpoints/001')