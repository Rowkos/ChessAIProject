import copy

import numpy as np
import pygame

pygame.init()
width, height = (600, 600)
squareWidth, squareHeight = (width / 8, height / 8)
window = pygame.display.set_mode((width, height))
timer = pygame.time.Clock()

darkColor = (239, 215, 191)
lightColor = (168, 121, 100)
highlight_color = (255, 0, 0)

directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]

def drawSquare(color, position):
    pygame.draw.rect(window, color, [position[0], position[1], squareWidth, squareHeight])

class Board:
    def __init__(self, window):
        self.window = window
        self.board_state = np.zeros((8, 8))
        self.piece_type_names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        self.piece_to_type_dict = {"empty": 0, "pawn": 1, "knight": 2, "bishop": 3, "rook": 4, "queen": 5, "king": 6}
        self.suits = ["white", "black"]
        self.squares_for_en_passant = [[], []]
        self.can_castle = [(True, True), (True, True)] # queens side, kings side

        self.sprites = []
        for i in range(12):
            # pawn knight bishop rook queen king
            img = pygame.image.load(r"C:/Users/rowan/PycharmProjects/Sentence-Correctness-Analysis-v2/"
                                    r"chess_piece_images/" + self.piece_type_names[i // 2] + "_" + self.suits[i % 2] + ".png"). \
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
            target_square = (origin[0], origin[1] + direction_of_movement * 2)
            if self.in_bounds(target_square):
                if self.board_state[target_square] == 0 and ((suit_of_piece == 0 and origin[1] == 6) or
                                                             (suit_of_piece == 1 and origin[1] == 1)):
                    if check_for_check:
                        if not self.sim_board_for_check(origin, target_square):
                            moves.append(target_square)
                    else:
                        moves.append(target_square)

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

    def get_all_possible_moves_by_suit(self, suit, only_attack = False):
        moves = []
        for x in range(8):
            for y in range(8):
                origin = (x, y)
                piece_on_origin = self.board_state[origin]
                if piece_on_origin != 0 and self.get_suit(piece_on_origin) == suit and self.get_piece_type(piece_on_origin) != 6:
                    moves.append(self.get_possible_moves(origin, only_attack = only_attack))
        # clean up
        moves = [g for g in moves if g != []]
        new_moves = []
        for x in moves:
            for y in x:
                new_moves.append(y)
        moves = new_moves
        return moves

    def king_is_in_check(self, suit_of_king):
        king_position = self.find_piece(self.get_piece_id("king", suit_of_king))
        king = self.board_state[king_position]
        suit_of_piece_on_origin = self.get_suit(king)
        enemy_suit = 1 if suit_of_piece_on_origin == 0 else 0
        if king_position in self.get_all_possible_moves_by_suit(enemy_suit):
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
            possible_moves = self.get_all_possible_moves_by_suit(suit) + self.get_possible_moves(self.find_piece(self.get_piece_id("king", suit)))

            # find an empty cell
            original_board_state = copy.deepcopy(self.board_state)
            empty_cell = 0
            for x in range(8):
                for y in range(8):
                    if self.board_state[x,y] == 0:
                        empty_cell = (x,y)
            self.board_state[empty_cell] = suit + 1
            for move in possible_moves:
                if not self.sim_board_for_check(empty_cell, move):
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
            if origin[0] - target[0] == 2:
                self.board_state[(target[0] - 2, target[1])] = 0
                self.board_state[(origin[0] - 1, origin[1])] = self.get_piece_id("rook", suit_of_piece_on_origin)