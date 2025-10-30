// src/components/GameSelect.js
import React from "react";
import "./GameSelect.css";

export default function GameSelect({ onSelect }) {
  return (
    <div className="select-container">
      <h1 className="select-title">Choose Your Game</h1>
      <div className="button-container">
        <button onClick={() => onSelect("tictactoe")} className="game-button">
          Tic Tac Toe
        </button>
        <button onClick={() => onSelect("connect4")} className="game-button">
          Connect 4
        </button>
      </div>
    </div>
  );
}
