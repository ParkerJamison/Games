import torch
import torch.nn as nn
import random
from TTT import TicTacToe  # <-- make sure this points to your TicTacToe class file


# -----------------------------------------------------
# Neural Net Definition (must match the one used in training)
# -----------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------
def encode_state(state):
    """Convert board symbols to numeric tensor for model."""
    mapping = {"X": 1.0, "O": -1.0, "-": 0.0}
    arr = [mapping[v] for v in state]
    return torch.FloatTensor(arr).unsqueeze(0)


def ai_move(game, model):
    """Use the trained model to choose a move."""
    state = encode_state(game.state)
    qvals = model(state)
    valid_moves = game.getValidMoves()

    # mask invalid moves
    mask = torch.tensor([float(i in valid_moves) for i in range(9)])
    qvals = qvals - (1 - mask) * 1e6

    # pick best valid move
    action = torch.argmax(qvals).item()
    return action


def print_board(state):
    """Nicely display the board in 3x3 grid."""
    print()
    for i in range(0, 9, 3):
        print(" | ".join(state[i:i+3]))
        if i < 6:
            print("---------")
    print()


# -----------------------------------------------------
# Game Loop
# -----------------------------------------------------
def play_game(model, human_player="O"):
    """Play a human vs AI or AI vs random game."""
    game = TicTacToe()
    current = "X"  # AI always starts as X

    while True:
        print_board(game.state)

        # check game status
        winner = game.checkWin()
        if winner:
            print(f"🏁 Winner: {winner}")
            break

        valid_moves = game.getValidMoves()
        if not valid_moves:
            print("🤝 Draw!")
            break

        if current == "X":  # AI turn
            move = ai_move(game, model)
            print(f"🤖 AI ({current}) chooses position {move}")
        else:
            if human_player == "O":
                # human input
                move = int(input("Your move (0–8): "))
                if move not in valid_moves:
                    print("❌ Invalid move. Try again.")
                    continue
            else:
                # random opponent
                move = random.choice(valid_moves)
                print(f"🎲 Random ({current}) chooses {move}")

        game = game.makeMove(move)
        current = "O" if current == "X" else "X"


# -----------------------------------------------------
# Main Entry Point
# -----------------------------------------------------
if __name__ == "__main__":
    # Load trained model
    model = Net()
    model.load_state_dict(torch.load("tictactoe_weights.pth", map_location=torch.device("cpu")))
    model.eval()

    print("✅ Loaded trained TicTacToe model.")
    print("Type '1' to play against AI, or '2' to watch AI vs random.")
    choice = input("> ")

    if choice.strip() == "1":
        play_game(model, human_player="O")
    else:
        play_game(model, human_player=None)
