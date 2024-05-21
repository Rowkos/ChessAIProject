import chess.pgn
from Board import *

pgn = open("lichess_db_standard_rated_2024-01.pgn", encoding="utf-8")
game_files = []

num_games = 1
for i in range(num_games):
    game = chess.pgn.read_game(pgn)
    game_files.append(game)
print(game_files[0].next().board())
print(str(game_files[0].next().board()).split("\n"))
print([x.board() for x in game_files[0].next().variations])

board = Board(window)
board.set_board_state_from_chess_game(game_files[0])

