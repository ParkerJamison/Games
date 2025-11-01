"""



"""


class Connect4():

    def __init__(self, state=None, currentPlayer="X", width=7, height=6, numMatch=4):
        self.WIDTH = width
        self.HEIGHT = height
        self.NUM_MATCH = numMatch
        if state == None:
            self.state = [["-" for i in range(self.WIDTH)] for i in range(self.HEIGHT)]
        else:
            self.state = state

        self.currentPlayer = currentPlayer

    def makeMove(self, pos:int):

        if (pos < 0) or (pos >= self.WIDTH):
            raise ValueError("Invalid Move: Out of Column Range")
        if self.state[0][pos] != "-":
            raise ValueError("Invalid Move: Column is FULL")
        
        nextState = self.state.copy()
        for i in range(self.HEIGHT-1, -1, -1): # start at the bottom row, go up 1 row each loop and place a piece once the spot is open
            if nextState[i][pos] == "-":
                nextState[i][pos] = self.currentPlayer
                break
        nextPlayer = "O" if self.currentPlayer == "X" else "X"

        return Connect4(nextState, nextPlayer)
        

    def checkWin(self):
        if "-" not in self.state[0]:
            return "draw"

        HEIGHT = self.HEIGHT
        WIDTH = self.WIDTH 
        NUM_MATCH = self.NUM_MATCH
        for i in range(HEIGHT-1, 0, -1):
            for j in range(WIDTH):
                if self.state[i][j] == "-":
                    continue
                currPlayer = self.state[i][j]

                if ((j + (NUM_MATCH-1)) < WIDTH): 
                    for x in range(1, NUM_MATCH, 1):
                        if self.state[i][j+x] != currPlayer:
                            break
                    else:
                        return currPlayer
                if ((i - (NUM_MATCH+1)) >= 0):

                    for x in range(1, NUM_MATCH, 1):
                        if self.state[i-x][j] != currPlayer:
                            break
                    else:
                        return currPlayer
                    
                    if ((j + (NUM_MATCH-1)) < WIDTH):
                        for x in range(1, NUM_MATCH, 1):
                            if self.state[i-x][j+x] != currPlayer:
                                break
                        else:
                            return currPlayer
                    if ((j - (NUM_MATCH+1)) >= 0):
                        for x in range(1, NUM_MATCH, 1):
                            if self.state[i-x][j-x] != currPlayer:
                                break
                        else:
                            return currPlayer
        return None
    
    def getvalidMoves(self):
        return [i for i in range(self.WIDTH) if self.state[0][i] == "-"]

    def __str__(self):
        # board as a string
        return  "\n".join([" ".join(s) for s in self.state])







def main():
    # Simple command-line test game.
    game = Connect4()
    print("Player 1 = X, Player 2 = O")
    print("Enter moves as a single integer 0-7")
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