import numpy as np
import pygame
from Board import Board
from Board import *

promotionBoxWidth = 400
promotionBoxHeight = 100
padding_factor = (promotionBoxWidth - 75) / 3

def createBoard():
    for file in range(8):
        for rank in range(8):
            isLightSquare = (file + rank) % 2 != 0
            squareColor = lightColor if isLightSquare else darkColor
            squarePosition = (file * 75, rank * 75)
            drawSquare(squareColor, squarePosition)

def initiate_pieces():
    # rooks
    board.board_state[0, 0] = board.get_piece_id("rook", 1)
    board.board_state[7, 0] = board.get_piece_id("rook", 1)
    board.board_state[0, 7] = board.get_piece_id("rook", 0)
    board.board_state[7, 7] = board.get_piece_id("rook", 0)

    # knights
    board.board_state[1, 0] = board.get_piece_id("knight", 1)
    board.board_state[6, 0] = board.get_piece_id("knight", 1)
    board.board_state[1, 7] = board.get_piece_id("knight", 0)
    board.board_state[6, 7] = board.get_piece_id("knight", 0)

    # bishops
    board.board_state[2, 0] = board.get_piece_id("bishop", 1)
    board.board_state[5, 0] = board.get_piece_id("bishop", 1)
    board.board_state[2, 7] = board.get_piece_id("bishop", 0)
    board.board_state[5, 7] = board.get_piece_id("bishop", 0)

    # queen and king
    board.board_state[3, 0] = board.get_piece_id("queen", 1)
    board.board_state[4, 0] = board.get_piece_id("king", 1)
    board.board_state[3, 7] = board.get_piece_id("queen", 0)
    board.board_state[4, 7] = board.get_piece_id("king", 0)

    board.board_state[:, 1] = board.get_piece_id("pawn", 1)
    board.board_state[:, 6] = board.get_piece_id("pawn", 0)

def draw_promotion_choice(suit):
    pygame.draw.rect(window, (255, 255, 255, 100), [width / 2 - promotionBoxWidth / 2,
                                              height / 2 - promotionBoxHeight / 2,
                                              promotionBoxWidth, promotionBoxHeight])
    pieces_to_draw = [board.sprites[board.get_piece_id("knight", suit) - 1],
                      board.sprites[board.get_piece_id("bishop", suit) - 1],
                      board.sprites[board.get_piece_id("rook", suit) - 1],
                      board.sprites[board.get_piece_id("queen", suit) - 1]]
    start_position = width / 2 - promotionBoxWidth / 2
    for i in range(len(pieces_to_draw)):
        window.blit(pieces_to_draw[i], (start_position, height / 2 - 25))
        if len(promotion_boxes) < 4:
            promotion_boxes.append((start_position, height / 2 - 25))
        start_position += padding_factor


if __name__ == "__main__":
    start_position = []
    board = Board(window)
    initiate_pieces()
    print(board.get_all_possible_moves_by_suit(0))

    selected_square = None
    is_promoting = False
    promoting_square = ()
    turn = 0
    promotion_boxes = []
    # I added a ghost type for representing the en pessant moves

    while True:
        createBoard()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if not is_promoting:
                    clicked_square = (mouse_pos[0] // 75, mouse_pos[1] // 75)

                    if selected_square is not None and clicked_square in board.get_possible_moves(selected_square, check_for_check = True):
                        board.check_for_castling(selected_square, clicked_square)
                        board.move_piece(selected_square, clicked_square)
                        board.check_for_en_pessant_opening(selected_square, clicked_square)
                        if board.check_for_promotion(clicked_square):
                            is_promoting = True
                            promoting_square = clicked_square
                        selected_square = None
                        turn = 1 if turn == 0 else 0
                        if board.is_checkmate(turn):
                            enemy_turn = 1 if turn == 0 else 0
                            print("PLAYER " + str(enemy_turn) + " WINS!")
                        if board.is_stalemate(turn):
                            print("IT'S A DRAW")

                    if board.get_suit(board.board_state[clicked_square]) == turn:
                        selected_square = (mouse_pos[0] // 75, mouse_pos[1] // 75)
                else:
                    for i, box in enumerate(promotion_boxes):
                        # assuming width of 50
                        if (box[0] + 50) > mouse_pos[0] > (box[0]) and (box[1] + 50) > mouse_pos[1] > (box[1]):
                            board.board_state[promoting_square] = board.get_piece_id(["knight", "bishop", "rook", "queen"][i],
                                                                                     board.get_suit(board.board_state[promoting_square]))
                            is_promoting = False
                            promoting_square = ()

            if event.type == pygame.QUIT:
                pygame.quit()

        if selected_square is not None:
            board.highlight_squares(selected_square)

        board.draw_pieces_to_screen()
        if is_promoting:
            draw_promotion_choice(board.get_suit(board.board_state[promoting_square]))

        pygame.display.update()
        timer.tick(30)
    pygame.quit()
