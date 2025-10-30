export const WIDTH = 7
export const HEIGHT = 6
export const NUM_MATCH = 4

export function makeEmptyBoard() {
    let board = [];
    for (let i = 0; i < HEIGHT; i++) {
        let row = [];
        for (let j = 0; j < WIDTH; j++) {
            row.push("-");
        }
        board.push(row);
    }
    return board;
}

export function makeMove(board, pos, player) {
    if (pos < 0 || pos >= WIDTH) throw new Error("Invalid Move position: out of bounds");
    if (board[0][pos] !== "-") throw new Error("Invalid Move: Column is full");

    let nextState = structuredClone(board);
    for (let i = HEIGHT-1; i >= 0; i--) {
        if (nextState[i][pos] === "-") {
            nextState[i][pos] = player;
            break;
        }
    }
    let nextPlayer = player === "X" ? "O" : "X";
    return { nextState, nextPlayer, winner: checkWin(nextState)};
}

export function checkWin(board) {

    if (!board[0].includes("-")) {
        return "draw";
    }

    let matchCheck = NUM_MATCH - 1;


    for (let i = HEIGHT - 1; i >= 0; i--) {
        for (let j = 0; j < WIDTH; j++) {
            if (board[i][j] === "-") {
                continue;
            }
            let currPlayer = board[i][j];

            // check the row
            if ((j + matchCheck) < WIDTH) {
                let found = true;
                for (let x = 1; x < NUM_MATCH; x++) {
                    if (board[i][j + x] !== currPlayer) {
                        found = false;
                        break;
                    }
                }
                if (found) {
                    return currPlayer
                }
            }
            
            if ((i - matchCheck) >= 0) {
                { // check the col
                let found = true;
                for (let x = 1; x < NUM_MATCH; x++) {
                    if (board[i-x][j] !== currPlayer) {
                        found = false;
                        break;
                    }
                }
                if (found) {
                    return currPlayer
                }
                }

                // check the up right diagnol
                if ((j + matchCheck) < WIDTH) {
                    let found = true;
                    for (let x = 1; x < NUM_MATCH; x++) {
                        if (board[i-x][j+x] !== currPlayer) {
                            found = false;
                            break;
                        }
                    }
                    if (found) {
                        return currPlayer
                    }
                }
                
                // check the up left diagnol
                if ((j - matchCheck) >= 0) {
                    let found = true;
                    for (let x = 1; x < NUM_MATCH; x++) {
                        if (board[i-x][j-x] !== currPlayer) {
                            found = false;
                            break;
                        }
                    }
                    if (found) {
                        return currPlayer
                    }
                }
            }
        }
    }
    return null;
}

export function printBoard(board) {
    for (let i = 0; i < HEIGHT; i++) {
        console.log(board[i].join(" "))
    }
}

// let state = makeEmptyBoard();
// let player = "X";
// let win = null;
// ({nextState: state, nextPlayer: player, winner: win} = makeMove(state, 2, "X"));
// ({nextState: state, nextPlayer: player, winner: win} = makeMove(state, 2, "X"));
// 
// console.log(win);
// ({nextState: state, nextPlayer: player, winner: win} = makeMove(state, 2, "X"));
// ({nextState: state, nextPlayer: player, winner: win} = makeMove(state, 2, "X"));
// 
// let test = checkWin(state);
// console.log(test);
// printBoard(state);