

export function makeEmptyBoard() {
    return ["-", "-", "-", "-", "-", "-", "-", "-", "-"];
}

export function makeMove(board, pos, player) {

    if (pos < 0 || pos>= 9) throw new Error("Invalid Position");
    if (board[pos] !== "-") throw new Error("Position already taken");

    let newBoard = [...board];
    newBoard[pos] = player;

    let nextPlayer = player === "X" ? "O" : "X";

    return {newBoard, nextPlayer, winner: checkWin(newBoard)}
}

export function checkWin(board) {
    const WINNERS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],   // row
        [0, 3, 6], [1, 4, 7], [2, 5, 8],   //cols
        [0, 4, 8], [2, 4, 6]               //diagnols
        ]

    for (const [a, b, c] of WINNERS) {
        if (board[a] !== "-" && board[a] === board[b] && board[a] === board[c]) {
            return board[a];
        }
    }
    return board.includes("-") ? null : "draw";
}