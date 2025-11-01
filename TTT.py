"""
Tic Tac Toe Game Engine

"""


from typing import List

class TicTacToe:
    """
    Tic Tac Toe game engine used by the backend and LLM agent. 

    BOARD POSITIONS

        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8
            
    Game State represented as a list of strings. 
        "-" represents an open square
        "X" represents player X
        "O" represents player O
    
    Created with an immutable game state approach, each instance encapsulates a single game state and the current players turn.

    Attributes
    ----------
    state : List(str)
        A list of 9 strings representing the board,
        Each character is one of "X", "O", "-"

        Example: [X, O, -, X, -, O, -, -, -]
    currentPlayer : str
        The symbol ('X' or 'O') whose turn it is to play.

    Methods
    -------
    make_move(pos):
        Returns a new TicTacToe object with the move applied for the current player.
        Raises ValueError if the position is invalid or already occupied.
    checkWin():
        Checks if there is a winner or draw.
        Returns 'X', 'O', 'draw', or None if the game is still ongoing.
    getValidMoves():
        Returns a list of integer positions (0-8) that are empty and playable.
    """
    def __init__(self, state=["-", "-", "-", "-", "-", "-", "-", "-", "-"], currentPlayer="X"):
        self.state = state
        self.currentPlayer = currentPlayer


    def makeMove(self, pos: int):
        """
        BOARD POSITIONS

        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8

        Returns a new TicTacToe object if the move is valid. 
        """

        # If the game is over, or the move is not in the board, invalid move, return false
        if (pos < 0) or (pos >= 9):
            raise ValueError("Invalid Move: Game is over or move out of bounds")
        # If the requested move is not an empty spot on the board, invalid move, return false
        if self.state[pos] != "-":
            raise ValueError("Invalid Move: Move is already taken")

        nextState = self.state.copy()
        nextState[pos] = self.currentPlayer  # Update the game board

        nextPlayer = "O" if self.currentPlayer == "X" else "X"

        return TicTacToe(nextState, nextPlayer)

    def checkWin(self):
        WINNING_COMBOS = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
        (0, 4, 8), (2, 4, 6)               # diagonals
        ]
        for a, b, c in WINNING_COMBOS:
            if self.state[a] != "-" and self.state[a] == self.state[b] == self.state[c]:
                return self.state[a]
        if "-" not in self.state:
            return "draw"
        return None

    def getValidMoves(self) -> List[int]:
        return [pos for pos in range(9) if self.state[pos] == "-"]

    def __str__(self):
        return f"{self.state[0]} | {self.state[1]} | {self.state[2]}\n---------\n{self.state[3]} | {self.state[4]} | {self.state[5]}\n---------\n{self.state[6]} | {self.state[7]} | {self.state[8]}"
        


"""
def main():
    # Simple command-line test game.
    game = TicTacToe()
    print("Player 1 = X, Player 2 = O")
    print("Enter moves as a single integer 0-8")
    print(game)

    while not game.checkWin() in ["X", "O", "draw"]:
        print(f"\nPlayer {game.currentPlayer}'s turn.")
        move = input("Enter position: ").strip()

        if move.lower() in ["quit", "exit"]:
            print("Game exited.")
            break
        try:
            nextState = game.makeMove(int(move))
        except ValueError:
            print("Invalid move. Try again.")
            continue

        print("\nCurrent board:")
        print(game)
        print(game.getValidMoves())

        result = game.checkWin()
        if result is not None:
            if result == "draw":
                print("Game is a draw")
            else:
                print(f"\n🎉 Player {result} wins!")

        game = nextState



if __name__ == "__main__":
    main()
"""