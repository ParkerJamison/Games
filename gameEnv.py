import torch, torch.nn as nn
from typing import List
import random
import argparse
import importlib
import copy
from TTT import TicTacToe
from connect4 import Connect4
import json


class GameEnv():
    def __init__(self, game, players: List[str]):
        self.game = game
        self.players = players

    def reset(self):
        self.game = self.game.__class__()
        return self.game.state
    
    def step(self, action):
        try:
            nextState = self.game.makeMove(action)
            self.game = nextState

        except ValueError:
            return self.game, -10, True
        
        winner = self.game.checkWin()
        if winner is None:
            return self.game, 0, False
        
        if winner in self.players:
            return self.game, 1, True
        else: 
            return self.game, 0.5, True


    def viewState(self):
        print(self.game)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 9))
        
    def forward(self, x):
        return self.net(x)


def flattenState(state):
    if isinstance(state, list):
        return [cell for row in state for cell in row]
    else:
        return state

def main(gameLogic, gamePlayers: List[str]):

    gamma = 0.9
    epsilon = 1.0
    epsMin = 0.05
    epsDecay = 0.995
    lr = 1e-3
    episodes = 75000


    env = GameEnv(gameLogic, gamePlayers)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            fs = flattenState(state)
            symbol_to_num = {'X': 1.0, 'O': -1.0, '-': 0.0}
            fs = [symbol_to_num[v] for v in fs]
            s = torch.FloatTensor(fs).unsqueeze(0)
            if random.random() < epsilon:

                validMoves = env.game.getValidMoves()
                if len(validMoves) != 0:
                    action = random.choice(validMoves)
                else:
                    done = True
                    break
            else:
                qvals = model(s)
                mask = torch.tensor([float(v == 0) for v in fs])
                qvals = qvals - (1 - mask) * 1e6
                action = torch.argmax(qvals).item()

            nextState, reward, done = env.step(action)

            fs = flattenState(nextState.state)
            fs = [symbol_to_num[v] for v in fs]

            ns = torch.FloatTensor(fs).unsqueeze(0)

            qval = model(s)[0, action]

            with torch.no_grad():
                target = reward + gamma * model(ns).max().item() * (1 - done)

            loss = loss_fn(qval, torch.tensor(target))
            opt.zero_grad()
            loss.backward()
            opt.step()

            state = nextState.state

        epsilon = max(epsMin, epsilon * epsDecay)
        if (ep+1)%1000==0:
            print(f"Episode {ep+1}, epsilon={epsilon:.3f}")
            print(epsilon)

    torch.save(model.state_dict(), "tictactoe_weights.pth")
    weights = {k: v.tolist() for k,v in model.state_dict().items()}
    json.dump(weights, open("tictactoe_weights.json","w"))



def load_class(module_name: str, class_name: str):
    """Dynamically import a class from a module"""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, required=True,
                        help="Module name (e.g. environments.tictactoe)")
    parser.add_argument("--player", type=str, required=True,
                        help="Game State Player Representation. What is returned to dictate what player won. (X,O)")
    parser.add_argument("--env", type=str, required=True,
                        help="Name of the environment class")
    
    args = parser.parse_args()

    players = args.player.split(",")
    
    env_class = load_class(args.module, args.env)
    env = env_class()

    print(f"Created: {env}")
    main(env, players)




# test = GameEnv(TicTacToe(), ["X", "O"])
# 
# x, y, z = test.step(2)
# x, y, z = test.step(2)
# 
# x, y, z = test.step(1)
# x, y, z = test.step(5)
# x, y, z = test.step(0)



        

