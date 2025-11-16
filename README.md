# ChessHacks Submission

This is a bot submission for chesshacks! I implemented a neural networking using a policy & value head, trained on a kaggle dataset of Magnus Carlsen games.

# AI PORTION

The policy network receives a chess position & predicts which move magnus would play / has played in the past.The value network receives a chess position & evaluates how good the position is (between -1 to 1).

Both the networks are CNNs meaning they learn best from local patterns & combine these local patterns into higher-level concepts (tactics, king's safety, pawn structure).

I chose a dataset of games played by Magnus Carlsen because his decision implicitly involve pattern recognition & chess fundamentals, meaning the NN would not have to be trained on that explicitly.

## Layers used

1. Conv2d -> takes the 18 input pieces & learns feature maps
2. BatcNorm2d -> normalize activations within a batch (makes it more stable)
3. ReLu -> non-linearity (standard)
4. Residual Blocks -> stacks of Conv + BatchNorm + ReLu; this is a deep feature extractor

## Loss functions

We use CrossEntropyLoss for the policy network which is standard for classification problems, which is essentially what this network is (classifying moves based on 'what move would Carlsen play in this position').

For the value network, we use MSELoss. The value network is a regression problem (where we want to evaluate a position's score on a continuous scale).

Both loss metrics are used in the back propagation process.

# Search Function & Evaluation

After loading saved weights, we use the value network as an evaluation function combined w/ a negamax search (w/ alpha-beta pruning to save time).

Depth is set to 3 as to not bloat the program & result in slow response times (with consideration to hackathon restraints).

The policy network is run on the current board to get a prediction on how likely Carlsen is to play a move according to the network.

The policy network is used to sort legal moves by their policy probability (essentially 'search moves that Magnus likes' first) & the negamax + value network is used to score these moves.
