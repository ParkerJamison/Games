import React, { useState } from "react";
import "../App.css";

export default function Connect4({ onBack }) {
  const [board, setBoard] = useState(
    Array.from({ length: 6 }, () => Array(7).fill("-"))
  );
  const [currentPlayer, setCurrentPlayer] = useState("X");
  const [winner, setWinner] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleMove(col) {
    if (winner || loading) return;
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5050/api/connect4/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ col }), // backend expects column index
      });

      const data = await response.json();

      if (data.error) {
        console.error(data.error);
        return;
      }

      setBoard(data.board); // board is a list of lists
      setCurrentPlayer(data.currentPlayer);
      setWinner(data.winner);
    } catch (err) {
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  }

  async function resetBoard() {
    try {
      await fetch("http://localhost:5050/api/connect4/reset", { method: "POST" });
    } catch (err) {
      console.error("Error resetting:", err);
    }

    setBoard(Array.from({ length: 6 }, () => Array(7).fill("-")));
    setCurrentPlayer("X");
    setWinner(null);
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
