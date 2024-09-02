import random
class Puzzle: 
    GOAL_STATE = "0 1 2 3 4 5 6 7 8"
    INITIAL_STATE = "7 2 4 5 0 6 8 3 1"

    def __init__(self):
        self.goalState = self.GOAL_STATE.split()
        self.curState = self.INITIAL_STATE.split()
    
    def setSeed(self, seed_value):
        random.seed(seed_value)

    def setState(self, state):
        state_parts = state.split()
        
        if len(state_parts) != 9 or not all(part.isdigit() and 0 <= int(part) <= 8 for part in state_parts):
            print("Error: invalid puzzle state")
            return
        
        state_list = list(map(int, state_parts))
        
        if sorted(state_list) != list(range(9)):
            print("Error: invalid puzzle state")
            return
        
        self.curState = [[0 for _ in range(3)] for _ in range(3)]

        index = 0 
        for i in range(3):
            for j in range(3):
                if state_list[index] == 0: 
                    self.curState[i][j] = " "
                else: 
                    self.curState[i][j] = state_list[index]
                index += 1

        print("New state set:")
        self.printState()


    def printState(self):
        for row in self.curState:
            print(" ".join(map(str, row)))
        print("")

    def moveDirection(self, direction):
        if not self.validMove(direction):
            print(f"Error invalid command: Move {direction} is not valid.")
            return

        blank_i, blank_j = self.findBlank()
        
        if direction == "left":
            self.curState[blank_i][blank_j], self.curState[blank_i][blank_j - 1] = self.curState[blank_i][blank_j - 1], self.curState[blank_i][blank_j]
        
        elif direction == "right":
            self.curState[blank_i][blank_j], self.curState[blank_i][blank_j + 1] = self.curState[blank_i][blank_j + 1], self.curState[blank_i][blank_j]
        
        elif direction == "up":
            self.curState[blank_i][blank_j], self.curState[blank_i - 1][blank_j] = self.curState[blank_i - 1][blank_j], self.curState[blank_i][blank_j]
        
        elif direction == "down":
            self.curState[blank_i][blank_j], self.curState[blank_i + 1][blank_j] = self.curState[blank_i + 1][blank_j], self.curState[blank_i][blank_j]

        print(f"Moved {direction}:")
        self.printState()
    
    def findBlank(self):
        blank_position = None
        for i in range(3):
            for j in range(3):
                if self.curState[i][j] == " ":
                    blank_position = (i, j)
                    break
        blank_i, blank_j  = blank_position
        return blank_i, blank_j           


    def validMove(self, direction):
    # blank i row blanj j column
        blank_i, blank_j = self.findBlank()

        if direction == "up":
            return blank_i > 0
        elif direction == "down":
            return blank_i < 2
        elif direction == "left":
            return blank_j > 0
        elif direction == "right":
            return blank_j < 2
        else:
            return False

    def scrambleState(self, n):
        for _ in range(n): 
            valid_directions = []
            for direction in ["left", "right", "up", "down"]:
                if self.validMove(direction):
                    valid_directions.append(direction)
            
            if valid_directions:
                move = random.choice(valid_directions)
                print(f"Scramble move: {move}")  
                self.moveDirection(move)


def cmd(command_str, puzzle):
    command_str = command_str.strip()
    
    if not command_str or command_str.startswith("#") or command_str.startswith("//"):
        return
    
    parts = command_str.split()
    command = parts[0]
    
    try:
        if command == "setState":
            state = " ".join(parts[1:])
            puzzle.setState(state)
        elif command == "printState":
            puzzle.printState()
        elif command == "move":
            direction = parts[1]
            puzzle.moveDirection(direction)
        elif command == "scrambleState":
            n = int(parts[1])
            puzzle.scrambleState(n)
        elif command == "setSeed":
            seed_value = int(parts[1])
            puzzle.setSeed(seed_value)
        else:
            print(f"Error: invalid command: {command_str}")
    except IndexError:
        print(f"Error: invalid command: {command_str}")

def cmdfile(filename):
    puzzle = Puzzle()  
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):  
            line = line.strip()
            if line and not line.startswith("#"):  
                print(f"Executing command from line {line_num}: {line}")
            cmd(line, puzzle)

def cmd_interact():
    puzzle = Puzzle()  
    print("Welcome to the Eight Puzzle. Type 'exit' to quit.")
    
    while True:
        command_str = input("Enter a command: ").strip()
        
        if command_str.lower() == "exit":
            print("Exiting interactive mode.")
            break
        
        if command_str.startswith("#") or command_str.startswith("//") or not command_str:
            continue
        
        cmd(command_str, puzzle)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmdfile(sys.argv[1])  
    else:
        cmd_interact()  


