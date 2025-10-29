import React, { useState } from "react";
import "../App.css"; // reuse your existing dark theme styles

export default function TicTacToe({ onBack }) {
  const [board, setBoard] = useState([
    ["-", "-", "-"],
    ["-", "-", "-"],
    ["-", "-", "-"],
  ]);
  const [currentPlayer, setCurrentPlayer] = useState("X");
  const [winner, setWinner] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleClick(pos) {
    if (winner || loading) return;
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5050/api/tictactoe/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pos }),
      });
      const data = await response.json();
      if (data.error) {
        console.error(data.error);
        return;
      }
      setBoard([
        data.board.slice(0, 3),
        data.board.slice(3, 6),
        data.board.slice(6, 9),
      ]);
      setWinner(data.winner);
      setCurrentPlayer(data.currentPlayer);
    } catch (err) {
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  }

  async function resetBoard() {
    try {
      await fetch("http://localhost:5050/api/tictactoe/reset", {
        method: "POST",
      });
    } catch (err) {
      console.error("Error resetting:", err);
    }
    setBoard([
      ["-", "-", "-"],
      ["-", "-", "-"],
      ["-", "-", "-"],
    ]);
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
        {board.map((row, rIndex) =>
          row.map((cell, cIndex) => {
            const pos = rIndex * 3 + cIndex;
            return (
              <button
                key={pos}
                onClick={() => handleClick(pos)}
                className={`cell ${
                  cell === "X" ? "x-cell" : cell === "O" ? "o-cell" : ""
                }`}
              >
                {cell === "-" ? "" : cell}
              </button>
            );
          })
        )}
      </div>

      <button onClick={resetBoard} className="reset-btn">
        Reset Game
      </button>
    </div>
  );
}
