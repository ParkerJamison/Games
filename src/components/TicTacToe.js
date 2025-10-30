import React, { useState } from "react";
import "../App.css"; // reuse your existing dark theme styles
import { makeEmptyBoard, makeMove, checkWin } from "../logic/TTT";

export default function TicTacToe({ onBack }) {
  const [board, setBoard] = useState(makeEmptyBoard());
  const [currentPlayer, setCurrentPlayer] = useState("X");
  const [winner, setWinner] = useState(null);


  async function handleClick(pos) {
    if (winner || board[pos] !== "-") return;

    const { newBoard, nextPlayer, winner: w } = makeMove(board, pos, currentPlayer);
    setBoard(newBoard);
    setCurrentPlayer(nextPlayer);
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

      <h1 className="title">Tic Tac Toe</h1>
      {winner ? (
        <h2 className="subtitle">
          {winner === "Draw" ? "It's a Draw!" : `Winner: ${winner}`}
        </h2>
      ) : (
        <h2 className="subtitle">Player: {currentPlayer}</h2>
      )}
      <div className="board">
        {Array.from({ length: 3 }, (_, r) => (
          <div key={r} className="row">
            {board.slice(r * 3, r * 3 + 3).map((cell, c) => {
              const pos = r * 3 + c;
              return (
                <button
                  key={pos}
                  onClick={() => handleClick(pos)}
                  className={`cell ${cell === "X" ? "x-cell" : cell === "O" ? "o-cell" : ""}`}
                >
                  {cell === "-" ? "" : cell}
                </button>
              );
            })}
          </div>
        ))}
      </div>

      <button onClick={resetBoard} className="reset-btn">
        Reset Game
      </button>
    </div>
  );
}
