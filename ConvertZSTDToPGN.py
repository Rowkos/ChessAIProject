# here's the plan. the model gets inputs in the form of 12 bitboards.
# the model outputs a valuation for the game.

# load dataset from lichess
# create a board out of each state of the game. label it win, lose, or draw
#
import zstandard as zstd
zst_file_path = 'lichess_db_standard_rated_2013-01 (1).pgn.zst'
# Path to the decompressed output file
pgn_file_path = 'lichess_db_standard_rated_2024-01.pgn'

# Decompressing the .zst file
with open(zst_file_path, 'rb') as compressed_file, open(pgn_file_path, 'wb') as decompressed_file:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(compressed_file, decompressed_file)

