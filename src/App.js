import React, { useState } from "react";
import GameSelect from "./components/GameSelect";
import TicTacToe from "./components/TicTacToe";
import Connect4 from "./components/Connect4";

function App() {
  const [selectedGame, setSelectedGame] = useState(null);

  const handleGameSelect = (game) => {
    setSelectedGame(game);
  };

  const handleBack = () => {
    setSelectedGame(null);
  };

  return (
    <div className="App">
      {!selectedGame && <GameSelect onSelect={handleGameSelect} />}
      {selectedGame === "tictactoe" && <TicTacToe onBack={handleBack} />}
      {selectedGame === "connect4" && <Connect4 onBack={handleBack} />}
    </div>
  );
}

export default App;
