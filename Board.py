import copy

import numpy as np
import pygame
import tensorflow as tf

pygame.init()
width, height = (600, 600)
squareWidth, squareHeight = (width / 8, height / 8)
window = pygame.display.set_mode((width, height))
timer = pygame.time.Clock()

darkColor = (239, 215, 191)
lightColor = (168, 121, 100)
highlight_color = (255, 0, 0)

directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
def build_eval_model():
    class Evaluator(tf.keras.Model):
        def __init__(self):
            super(Evaluator, self).__init__(name = "Evaluator")
            self.flatten = tf.keras.layers.Flatten()
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
            x = self.flatten(x)
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
def drawSquare(color, position):
    pygame.draw.rect(window, color, [position[0], position[1], squareWidth, squareHeight])

class Board:
    def __init__(self, window):
        self.window = window
        self.model = build_eval_model()
        self.model(np.zeros(shape = (1, 768)))
        self.model.load_weights('ChessAICheckpoints02/001/001')
        self.model.save("Evaluator.keras")
        self.board_state = np.zeros((8, 8))
        self.piece_type_names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        self.piece_to_type_dict = {"empty": 0, "pawn": 1, "knight": 2, "bishop": 3, "rook": 4, "queen": 5, "king": 6}
        self.suits = ["white", "black"]
        self.squares_for_en_passant = [[], []]
        self.can_castle = [(True, True), (True, True)] # queens side, kings side

        self.sprites = []
        for i in range(12):
            # pawn knight bishop rook queen king
            img = pygame.image.load(r"pieces/" + self.suits[i % 2] + "_" + self.piece_type_names[i // 2] + ".png"). \
                convert_alpha()
            self.sprites.append(img)

    def get_piece_id(self, name, suit):
        return self.piece_to_type_dict[name] * 2 + suit - 1

    def draw_pieces_to_screen(self):
        for file in range(8):
            for rank in range(8):
                if self.board_state[file][rank] != 0:
                    squarePosition = (file * 75 + 5, rank * 75 + 7)  # add nums to center
                    self.window.blit(self.sprites[int(self.board_state[file][rank]) - 1], squarePosition)

    def highlight_squares(self, origin):
        piece = self.board_state[origin]
        drawSquare(highlight_color, (origin[0] * 75, origin[1] * 75))

        if piece != 0:
            possible_moves = self.get_possible_moves(origin, check_for_check = True)
            for move in possible_moves:
                drawSquare(highlight_color, (move[0] * 75, move[1] * 75))
    def get_suit(self, piece_id):
        # 0 white 1 black
        if piece_id % 2 == 0:
            return 1
        return 0

    def get_piece_type(self, piece_id):
        return (piece_id + 1) // 2

    def get_sliding_moves(self, origin, piece_on_origin, type_of_piece, check_for_check = False):
        moves = []
        if type_of_piece == 3:
            direction_subset = directions[4:]
        elif type_of_piece == 4:
            direction_subset = directions[:4]
        else:
            direction_subset = directions

        for i in range(len(direction_subset)):
            target_square = (origin[0], origin[1])
            for j in range(7):
                target_square = (target_square[0] + direction_subset[i][0], target_square[1] + direction_subset[i][1])


                if not self.in_bounds(target_square):
                    break

                piece_on_target_square = self.board_state[target_square]

                # same suit means cannot capture
                if self.get_suit(piece_on_origin) == self.get_suit(piece_on_target_square) and piece_on_target_square != 0:
                    break
                if check_for_check:
                    if self.sim_board_for_check(origin, target_square):
                        continue
                moves.append(target_square)
                if piece_on_target_square != 0:
                    break
        return moves

    def get_pawn_moves(self, origin, only_attack = False, check_for_check = False):
        moves = []
        piece_on_origin = self.board_state[origin]
        suit_of_piece = self.get_suit(piece_on_origin)
        direction_of_movement = -1 if suit_of_piece == 0 else 1

        # add an only attack flag so that we can fetch non-attack moves for the king
        if not only_attack:
            # 1 space move
            target_square = (origin[0], origin[1] + direction_of_movement)

            if self.in_bounds(target_square):
                if self.board_state[target_square] == 0:
                    if check_for_check:
                        if not self.sim_board_for_check(origin, target_square):
                            moves.append(target_square)
                    else:
                        moves.append(target_square)

            # 2 space move
            target_square_0 = (origin[0], origin[1] + direction_of_movement * 2)
            if self.in_bounds(target_square_0):
                if self.board_state[target_square_0] == 0 and self.board_state[target_square] == 0 and ((suit_of_piece == 0 and origin[1] == 6) or
                                                             (suit_of_piece == 1 and origin[1] == 1)):
                    if check_for_check:
                        if not self.sim_board_for_check(origin, target_square_0):
                            moves.append(target_square_0)
                    else:
                        moves.append(target_square_0)

        # capture and en pessant
        possible_capture_squares = [(origin[0] + 1, origin[1] + direction_of_movement),
                                    (origin[0] - 1, origin[1] + direction_of_movement)]
        for target_square in possible_capture_squares:
            if not self.in_bounds(target_square):
                continue

            # if only attack is on, we just want to see if the pawn can reach
            if only_attack:
                if check_for_check:
                    if self.sim_board_for_check(origin, target_square):
                        continue
                moves.append(target_square)
                continue

            if ((self.get_suit(self.board_state[target_square]) != suit_of_piece and self.board_state[target_square] != 0)
                    or (target_square in self.squares_for_en_passant[suit_of_piece] and self.board_state[target_square] == 0)):
                if check_for_check:
                    if self.sim_board_for_check(origin, target_square):
                        continue
                moves.append(target_square)

        return moves
    def get_knight_moves(self, origin, check_for_check = False):
        moves = []
        piece_on_origin = self.board_state[origin]
        directions_of_movement = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
        for direction in directions_of_movement:
            target_square = (origin[0] + direction[0], origin[1] + direction[1])

            if not self.in_bounds(target_square):
                continue

            piece_on_target_square = self.board_state[target_square]

            if piece_on_target_square == 0 or self.get_suit(piece_on_target_square) != self.get_suit(piece_on_origin):
                if check_for_check:
                    if self.sim_board_for_check(origin, target_square):
                        continue
                moves.append(target_square)

        return moves

    def get_king_moves(self, origin):
        moves = []
        piece_on_origin = self.board_state[origin]
        suit_of_piece_on_origin = self.get_suit(piece_on_origin)
        for direction in directions:
            target_square = (origin[0] + direction[0], origin[1] + direction[1])
            if not self.in_bounds(target_square):
                continue
            piece_on_target_square = self.board_state[target_square]

            # same suit means cannot capture
            if self.get_suit(piece_on_origin) == self.get_suit(piece_on_target_square) and piece_on_target_square != 0:
                continue

            if not self.sim_board_for_check(origin, target_square):
                moves.append(target_square)
        # castling queen side
        if (self.can_castle[suit_of_piece_on_origin][0] == True and not self.king_is_in_check(suit_of_piece_on_origin)
            and self.board_state[(origin[0] - 1, origin[1])] == 0
            and self.board_state[(origin[0] - 2, origin[1])] == 0
            and self.board_state[(origin[0] - 3, origin[1])] == 0):
            if (not self.sim_board_for_check(origin, (origin[0] - 1, origin[1])) and
               not self.sim_board_for_check(origin, (origin[0] - 2, origin[1]))):
                moves.append((origin[0] - 2, origin[1]))

        # castling kings side
        if (self.can_castle[suit_of_piece_on_origin][1] == True and not self.king_is_in_check(suit_of_piece_on_origin)
            and self.board_state[(origin[0] + 1, origin[1])] == 0
            and self.board_state[(origin[0] + 2, origin[1])] == 0):
            if (not self.sim_board_for_check(origin, (origin[0] + 1, origin[1])) and
               not self.sim_board_for_check(origin, (origin[0] + 2, origin[1]))):
                moves.append((origin[0] + 2, origin[1]))
        return moves

    def check_if_king_adjacent(self, origin, suit_to_look_for):
        adjacent_squares = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        for adjacent_square in adjacent_squares:
            square = (origin[0] + adjacent_square[0], origin[1] + adjacent_square[1])
            if self.in_bounds(square):
                if self.board_state[square] == self.get_piece_id("king", suit_to_look_for):
                    return True
        return False

    def get_all_possible_moves_by_suit(self, suit, only_attack = False, return_origins = False, check_for_check = False, test_flag = False, get_king_moves = False):
        moves = []
        origins = []
        for x in range(8):
            for y in range(8):
                origin = (x, y)
                piece_on_origin = self.board_state[origin]
                if piece_on_origin != 0 and self.get_suit(piece_on_origin) == suit and self.get_piece_type(piece_on_origin) != 6:
                    possible_moves = self.get_possible_moves(origin, only_attack = only_attack, check_for_check = check_for_check)
                    moves.append(possible_moves)
                    origins.append([origin] * len([x for x in possible_moves if x != []]))
        if get_king_moves:
            king_moves = self.get_possible_moves(self.find_piece(self.get_piece_id("king", suit)))
            moves.append(king_moves)
            origins.append([self.find_piece(self.get_piece_id("king", suit))] * len([x for x in king_moves if x != []]))
        # clean up
        moves = [g for g in moves if g != []]
        origins = [g for g in origins if g != []]
        new_moves = []
        new_origins = []
        for i, x in enumerate(moves):
            for j, y in enumerate(x):
                new_moves.append(y)
                new_origins.append(origins[i][j])
        moves = new_moves
        origins = new_origins
        if return_origins:
            return [moves, origins]
        return moves

    def king_is_in_check(self, suit_of_king):
        king_position = self.find_piece(self.get_piece_id("king", suit_of_king))
        king = self.board_state[king_position]
        suit_of_piece_on_origin = self.get_suit(king)
        enemy_suit = 1 if suit_of_piece_on_origin == 0 else 0
        if (king_position in self.get_all_possible_moves_by_suit(enemy_suit) or
                self.check_if_king_adjacent(king_position, enemy_suit)):
            return True

        return False

    def sim_board_for_check(self, origin, target_square):
        piece_on_origin = self.board_state[origin]
        current_board_state = copy.deepcopy(self.board_state)
        self.move_piece(origin, target_square)
        if self.king_is_in_check(self.get_suit(piece_on_origin)):
            self.board_state = current_board_state
            return True
        self.board_state = current_board_state
        return False

    def find_piece(self, piece_id):
        for x in range(8):
            for y in range(8):
                if self.board_state[x, y] == piece_id:
                    return x,y

    def get_possible_moves(self, origin, only_attack = False, check_for_check = False):
        piece_on_origin = self.board_state[origin]
        type_of_piece = self.get_piece_type(piece_on_origin)

        if type_of_piece == 6:
            return self.get_king_moves(origin)

        if type_of_piece == 3 or type_of_piece == 4 or type_of_piece == 5:
            return self.get_sliding_moves(origin, piece_on_origin, type_of_piece, check_for_check = check_for_check)
        if type_of_piece == 1:
            return self.get_pawn_moves(origin, only_attack, check_for_check = check_for_check)
        if type_of_piece == 2:
            return self.get_knight_moves(origin, check_for_check = check_for_check)
        return []

    def in_bounds(self, position):
        if position[0] < 0 or position[0] > 7 or position[1] < 0 or position[1] > 7:
            return False
        else:
            return True

    def move_piece(self, origin, target):
        piece_on_origin = self.board_state[origin]
        self.board_state[target] = piece_on_origin
        self.board_state[origin] = 0

        # check for capture en pessant
        if target in self.squares_for_en_passant[self.get_suit(piece_on_origin)]:
            self.board_state[(target[0], origin[1])] = 0

    def is_checkmate(self, suit):
        if self.king_is_in_check(suit):
            can_escape = False
            possible_moves = self.get_all_possible_moves_by_suit(suit, return_origins=True)
            king_moves = self.get_possible_moves(self.find_piece(self.get_piece_id("king", suit)))
            possible_moves[0] += king_moves
            possible_moves[1] += [self.find_piece(self.get_piece_id("king", suit))] * len(king_moves)
            # find an empty cell
            original_board_state = copy.deepcopy(self.board_state)
            for i in range(len(possible_moves[0])):
                if not self.sim_board_for_check(possible_moves[1][i], possible_moves[0][i]):
                    can_escape = True
            self.board_state = original_board_state
            return not can_escape
        return False

    def is_stalemate(self, suit):
        if not self.king_is_in_check(suit):
            can_escape = False
            possible_moves = self.get_all_possible_moves_by_suit(suit, return_origins=True, test_flag = True)
            king_moves = self.get_possible_moves(self.find_piece(self.get_piece_id("king", suit)))

            possible_moves[0] += king_moves
            possible_moves[1] += [self.find_piece(self.get_piece_id("king", suit))] * len(king_moves)
            # find an empty cell
            original_board_state = copy.deepcopy(self.board_state)

            for i in range(len(possible_moves[0])):
                if not self.sim_board_for_check(possible_moves[1][i], possible_moves[0][i]):
                    can_escape = True
            self.board_state = original_board_state
            return not can_escape
        return False
    def check_for_en_pessant_opening(self, origin, target):
        suit_of_origin = self.get_suit(self.board_state[target])
        enemy_suit = 1 if suit_of_origin == 0 else 0
        self.squares_for_en_passant[enemy_suit] = []
        if self.get_piece_type(self.board_state[target]) == 1 and abs(origin[1] - target[1]) == 2:
            self.squares_for_en_passant[enemy_suit].append((origin[0], (origin[1] + target[1]) / 2))

    def check_for_promotion(self, target):
        suit_of_origin = self.get_suit(self.board_state[target])
        if self.get_piece_type(self.board_state[target]) == 1 and ((suit_of_origin == 0 and target[1] == 0) or (suit_of_origin == 1 and target[1] == 7)):
            return True
        return False

    def check_for_castling(self, origin, target):
        piece_on_origin = self.board_state[origin]
        suit_of_piece_on_origin = self.get_suit(piece_on_origin)
        if piece_on_origin == self.get_piece_id("king", suit_of_piece_on_origin):
            # castling king's side
            if origin[0] - target[0] == -2:
                self.board_state[(target[0] + 1, target[1])] = 0
                self.board_state[(origin[0] + 1, origin[1])] = self.get_piece_id("rook", suit_of_piece_on_origin)
            elif origin[0] - target[0] == 2:
                self.board_state[(target[0] - 2, target[1])] = 0
                self.board_state[(origin[0] - 1, origin[1])] = self.get_piece_id("rook", suit_of_piece_on_origin)

            self.can_castle[suit_of_piece_on_origin] = (False, False)
        if piece_on_origin == self.get_piece_id("rook", suit_of_piece_on_origin):
            if origin[0] == 0 and (origin[1] == 0 or origin[1] == 7):
                self.can_castle[suit_of_piece_on_origin] = (False, self.can_castle[suit_of_piece_on_origin][1])

            if origin[0] == 7 and (origin[1] == 0 or origin[1] == 7):
                self.can_castle[suit_of_piece_on_origin] = (self.can_castle[suit_of_piece_on_origin][1], False)

    def board_to_list(self, board):
        rows = str(board).split("\n")
        return [x.split() for x in rows]

    def get_moved_piece(self, prev_state, new_state):
        origin = ()
        target = ()
        for x in range(8):
            for y in range(8):
                if prev_state[x][y] != new_state[x][y]:
                    if new_state[x][y] == ".":
                        # the piece left this square
                        origin = (x, y)
                    else:
                        target = (x, y)
        return origin, target

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
        prev_board = self.board_to_list(game.board())
        states = []
        while True:
            game = game.next()
            board_state = self.board_to_list(game.board())
            origin, target = self.get_moved_piece(prev_board, board_state)

            if game.is_end():
                break
            prev_board = copy.deepcopy(board_state)
            self.move_piece(origin, target)
            states.append(self.board_state)
        if return_intermediates:
            return states

    def set_board_state_from_list(self, li):
        for x in range(8):
            for y in range(8):
                self.board_state[x, y] = li[y][x]

    def rotate_board(self, board_state):
        new_board = np.zeros((8, 8))
        for x in range(8):
            for y in range(8):
                new_board[y][x] = board_state[x][y]
        return new_board

    def make_AI_move(self, suit):
        # let's try building a depth based search to go more moves into the future
        moves_and_origins = self.get_all_possible_moves_by_suit(suit, return_origins = True, check_for_check = True, get_king_moves = True)
        # need to get king moves
        possible_moves = moves_and_origins[0]
        origins = moves_and_origins[1]

        ratings = []
        other_suit = 1 if suit == 0 else 0
        for i in range(len(possible_moves)):
            prev_board_state = copy.deepcopy(self.board_state)
            prev_castling = copy.deepcopy(self.can_castle)
            self.check_for_castling(origins[i], possible_moves[i])
            self.move_piece(origins[i], possible_moves[i])
            ratings.append(self.depth_based_search(2, other_suit, suit))

            # state = tf.constant([self.get_bitboard(copy.deepcopy(self.board_state))])
            # ratings.append(self.model(state))
            self.board_state = prev_board_state
            self.can_castle = prev_castling

        print(ratings)
        selected_move = np.argmax(ratings)
        print(selected_move)
        self.check_for_castling(origins[selected_move], possible_moves[selected_move])
        self.move_piece(origins[selected_move], possible_moves[selected_move])

    def depth_based_search(self, depth, current_suit, player_suit, best_rating = None):
        if depth > 1:
            moves = self.get_all_possible_moves_by_suit(current_suit, return_origins = True, check_for_check = True, get_king_moves = True)
            ratings = []
            alt_suit = 0 if current_suit == 1 else 1
            for i in range(len(moves[0])):
                prev_board_state = copy.deepcopy(self.board_state)
                prev_castling = copy.deepcopy(self.can_castle)
                self.check_for_castling(moves[1][i], moves[0][i])
                self.move_piece(moves[1][i], moves[0][i])
                ratings += self.depth_based_search(depth - 1, alt_suit, player_suit)
                self.board_state = prev_board_state
                self.can_castle = prev_castling
            if current_suit == player_suit:
                return [max(ratings)]
            else:
                return [min(ratings)]
        else:
            state = tf.constant([self.get_bitboard(self.rotate_board(copy.deepcopy(self.board_state)))])
            if player_suit == 0:
                return [self.model(state).numpy().tolist()[0][0]]
            else:
                return [1 - self.model(state).numpy().tolist()[0][0]]

    def get_bitboard(self, state):
        bitboards = np.zeros((12, 8, 8))
        for x in range(8):
            for y in range(8):
                if state[x, y] != 0:
                    bitboards[int(state[x, y]) - 1, x, y] = 1
        return bitboards
