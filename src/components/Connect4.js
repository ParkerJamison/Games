import React, { useState } from "react";
import "../App.css";
import { makeEmptyBoard, makeMove } from "../logic/Connect4Logic";

export default function Connect4({ onBack }) {
  const [board, setBoard] = useState(makeEmptyBoard());
  const [currentPlayer, setCurrentPlayer] = useState("X");
  const [winner, setWinner] = useState(null);

  async function handleMove(pos) {
    if (winner || board[0][pos] !== "-") return;

    const { nextState: newBoard, nextPlayer: player , winner: w } = makeMove(board, pos, currentPlayer);
  
    setBoard(newBoard);
    setCurrentPlayer(player);
    setWinner(w);
  }

  async function resetBoard() {
    setBoard(makeEmptyBoard());
    setWinner(null);
    setCurrentPlayer("X");
  }

  return (
    <div className="app-container">
      <button onClick={onBack} className="back-button">← Back</button>

      <h1 className="title">Connect 4</h1>
      {winner ? (
        <h2 className="subtitle">
          {winner === "draw" ? "It's a Draw!" : `Winner: ${winner}`}
        </h2>
      ) : (
        <h2 className="subtitle">Player: {currentPlayer}</h2>
      )}

      {/* reverse() so bottom row renders visually at the bottom */}
      <div className="connect4-board">
        {[...board].map((row, rIndex) => (
          <div key={rIndex} className="connect4-row">
            {row.map((cell, cIndex) => (
              <div
                key={cIndex}
                className="connect4-cell"
                onClick={() => handleMove(cIndex)}
              >
                <div
                  className={`disc ${
                    cell === "-"
                      ? ""
                      : cell === "X"
                      ? "disc-red"
                      : "disc-yellow"
                  }`}
                />
              </div>
            ))}
          </div>
        ))}
      </div>

      <button onClick={resetBoard} className="reset-btn">
        Reset Game
      </button>
    </div>
  );
}
