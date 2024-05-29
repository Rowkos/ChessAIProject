# ChessAIProject
 

This project covers the whole life cycle of building an Artificial Intelligence to play the game of chess.
Data Collection

To begin with, we need to find data to train our model. The most readily available source of data is the Lichess database. Data from any month will work, but this projection utilizes part of the data from the most recent April 2024 shard. Initially, this data is in the form of a .zsdt file which is converted to a .pgn file. We then cut out all the games without evaluations, which is about 95% of them. 
Model Training 

You can train a model with any architecture you want, as long as it ends in a single sigmoid neuron. 

The model is trained by generating a dataset of positions along with an associated valuation and learning to predict that valuation. These valuations are computed painstakingly by the stockfish AI over many seconds but our model learns to do them much faster.

Playing the AI

You can then play against your own AI by saving it to a local directory and substituting in its URL for the model parameter of the board script.

The AI works by recursively generating every possible move and every possible response from your opponent up to a predefined depth (you can set this to whatever you want although it will get very slow very fast). The model scores each possible move between 0 and 1. 0 being black will definitely win and 1 being white will definitely win. We then use the minimax algorithm to determine the best move.

Compared to the traditional minimax algorithm that scores positions based on which pieces are left, we need to look through far fewer future moves to get a decent valuation.

Improvements

This is a pretty bad AI, to be honest. But it could be made better. First of all, implementing a transposition table and alpha-beta pruning could increase the distance into the future that we are able to see. Secondly, we could train a larger model, for longer, on more data. 

All in all, I feel it turned out well for the time available.
