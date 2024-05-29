# here's the plan. the model gets inputs in the form of 12 bitboards.
# the model outputs a valuation for the game.

# load dataset from lichess
# create a board out of each state of the game. label it win, lose, or draw
#
import zstandard as zstd
zst_file_path = r'C:\Users\rowan\Downloads\lichess_db_standard_rated_2024-04.pgn.zst'
# Path to the decompressed output file
pgn_file_path = r'C:\Users\rowan\Downloads\lichess_db_standard_rated_2024-04.pgn'

# Decompressing the .zst file
with open(zst_file_path, 'rb') as compressed_file, open(pgn_file_path, 'wb') as decompressed_file:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(compressed_file, decompressed_file)

