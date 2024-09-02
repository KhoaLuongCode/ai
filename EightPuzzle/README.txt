# Eight Puzzle Code Overview

## How to run code

### `interactive_cmd` and `cmdfile` func 
Enable command execution:
- Interactive mode: `python EightPuzzle.py` - type exit to exit the interactive mode.
- File-based execution: `python EightPuzzle.py testcmds.txt`

## Code Organization and Design Choices

### 1. `Puzzle` Class
Manages and encapsulates the puzzle's state, moves, and validations.

### 2. `setState`
Validates and updates the puzzle's state. Ensures input contains exactly nine numeric values, including 0 through 8.

### 3. `setSeed`
Sets the random seed to ensure reproducibility of scrambled puzzle states.

### 4. `cmd`
Executes commands by calling relevant `Puzzle` methods like `setState` and `moveDirection`. Includes error handling for invalid commands.

### 5. Error Handling
Embedded in functions like `setState` to ensure the program handles invalid inputs gracefully.

### 6. `findBlank`
Supports `moveDirection` and `validMove` by locating the blank space in the puzzle grid.

### 7. `moveDirection`
Executes valid moves based on input direction. Uses `findBlank` func to determine the blank tile's position and `validMove` func to determine valid move before moving.

### 8. `validMove`
Checks move validity and supports both `moveDirection` and `scrambleState` by ensuring moves are within grid boundaries. 

### 9. `scrambleState`
Randomizes the puzzle state by making a series of valid moves. Uses on `validMove` to ensure each move is legitimate - goes through all moves and determine valid ones.

## Examples of Valid Commands

- `setState 1 2 3 4 5 6 7 8 0` 
- `move up`
- `move left` 
- `scrambleState 10` 
- `setSeed 20` 
- `printState` 

