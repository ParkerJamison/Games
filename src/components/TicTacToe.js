import React, { useState, useEffect } from "react";
import "../App.css"; // reuse your existing dark theme styles
import { makeEmptyBoard, makeMove, checkWin } from "../logic/TTT";

async function loadWeights() {
  const res = await fetch(`${process.env.PUBLIC_URL}/tictactoe_weights.json`);

  if (!res.ok) throw new Error("Failed to load weights");
  return await res.json();
}


function relu(x) {
  return x.map(v => Math.max(0, v));
}

function matmul(a, b, bias) {

  const rows = a.length;
  const cols = b[0].length;
  const out = new Array(cols).fill(0);



  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) out[j] += a[i] * b[i][j];
    if (bias) out[j] += bias[j];
  }
  return out;
}

function transpose(m) {
  return m[0].map((_, i) => m.map(row => row[i]));
}


function forward(input, w) {
  // input is array of 9 floats (board state: X=1, O=-1, empty=0)
  let x = input;

  // Layer 1: Linear(9→64) + ReLU
  x = relu(matmul(x, transpose(w["net.0.weight"]), w["net.0.bias"]));

  // Layer 2: Linear(64→64) + ReLU
  x = relu(matmul(x, transpose(w["net.2.weight"]), w["net.2.bias"]));

  // Layer 3: Linear(64→9)
  x = matmul(x, transpose(w["net.4.weight"]), w["net.4.bias"]);

  return x; // Q-values for 9 positions
}

function chooseAction(board, weights) {
  const qValues = forward(board, weights);

  // Mask invalid moves (cells already filled)
  const validMoves = board.map((v, i) => (v === 0 ? qValues[i] : -Infinity));
  const bestIdx = validMoves.indexOf(Math.max(...validMoves));
  return bestIdx;
}

function encodeBoard(board, aiPlayer) {
  // board: array like ["X", "O", "-", "-", "X", "-", "-", "O", "-"]
  // aiPlayer: which side the model was trained as ("X" or "O")
  // returns numeric array like [1, -1, 0, 0, 1, 0, 0, -1, 0]

  return board.map(cell => {
    if (cell === "-") return 0;
    if (cell === aiPlayer) return -1;
    return 1;
  });
}


export default function TicTacToe({ onBack }) {
  const [board, setBoard] = useState(makeEmptyBoard());
  const [currentPlayer, setCurrentPlayer] = useState("X");
  const [winner, setWinner] = useState(null);
  const [aiWeights, setAIWeights] = useState(null);


  useEffect(() => {
  loadWeights().then(setAIWeights);
  }, []);

  async function handleClick(pos) {
    if (winner || board[pos] !== "-") return;

    const { newBoard, nextPlayer, winner: w } = makeMove(board, pos, currentPlayer);
    setBoard(newBoard);
    setCurrentPlayer(nextPlayer);
    setWinner(w);

    // If game not over, it's AI's turn (O)
    if (!w && nextPlayer === "O" && aiWeights) {
      setTimeout(() => aiMove(newBoard), 400); // delay for realism
    }
  }

  async function aiMove(currentBoard) {
    // Encode board for AI perspective (AI plays as "O")
    const encoded = encodeBoard(currentBoard, "O");

    console.log(encoded)
    const bestMove = chooseAction(encoded, aiWeights);

    console.log(bestMove);

    // Apply AI move
    const { newBoard, nextPlayer, winner: w } = makeMove(currentBoard, bestMove, "O");
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
