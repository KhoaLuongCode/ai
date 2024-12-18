Eight Puzzle Code Overview
==========================

How to Run Code: Command Execution
----------------------------------
Interactive Mode: 
    - Type exit to exit the interactive mode.
    python EightPuzzle.py

File-Based Execution: 
    - Testing state representation and basic functionalities:
    python EightPuzzle.py state.txt

    - Testing search algorithms (BFS/DFS):
    python EightPuzzle.py search.txt


Code Organization and Design Choices
------------------------------------

Puzzle Class
------------
The Puzzle class encapsulates the state of the puzzle and provides methods to manipulate it. This design ensures that all puzzle-related logic is self-contained, making the code easier to maintain and extend.

    class Puzzle: 
        MAX_NODES_LIMIT = 1000
        DEFAULT_DIRECTIONS = ["left", "right", "up", "down"]
        GOAL_STATE = "0 1 2 3 4 5 6 7 8"
        INITIAL_STATE = "7 2 4 5 0 6 8 3 1"


setState Method
---------------
The setState method validates the provided state to ensure it contains exactly nine numeric values, including 0 through 8. If the validation passes, it updates the current state.

    if len(state_parts) != 9 or not all(part.isdigit() and 0 <= int(part) <= 8 for part in state_parts):
        print("Error: invalid puzzle state")


setSeed Method
--------------
The setSeed method sets the random seed to ensure consistency when scrambling the puzzle.

    def setSeed(self, seed_value):
        random.seed(seed_value)


Error Handling
--------------
Error handling is embedded in methods like setState and cmd to ensure that invalid inputs are gracefully managed. For example, the setState method prints an error message if the state is invalid.

    if len(state_parts) != 9 or not all(part.isdigit() and 0 <= int(part) <= 8 for part in state_parts):
        print("Error: invalid puzzle state")
        return


findBlank Method
----------------
The findBlank method locates the blank space in the puzzle grid, which is crucial for determining valid moves. It returns the position of the blank tile.

    def findBlank(self):
        for i in range(3):
            for j in range(3):
                if self.curState[i][j] == " ":
                    return i, j


moveDirection Method
--------------------
The moveDirection method executes valid moves based on the input direction. It uses findBlank to determine the blank tile's position and validMove to validate the move before executing it.

    def moveDirection(self, direction, line_num=None):
        if not self.validMove(direction):
            print(f"Error: invalid command: Move {direction} is not valid on line {line_num}.")
            return
        ...


validMove Method
----------------
The validMove method checks the validity of a move. It supports both moveDirection and scrambleState by ensuring that moves stay within grid boundaries.

    def validMove(self, direction):
        blank_i, blank_j = self.findBlank()
        ...
        elif direction == "left":
            return blank_j > 0
        elif direction == "right":
            return blank_j < 2
        else:
            return False


scrambleState Method
--------------------
The scrambleState method randomizes the puzzle state by making a series of valid moves only. It uses validMove to ensure each move is legitimate.

    def scrambleState(self, n):
        for _ in range(n): 
            valid_directions = [direction for direction in ["left", "right", "up", "down"] if self.validMove(direction)]
            if valid_directions:
                move = random.choice(valid_directions)
                self.moveDirection(move)

BFS Search
--------------------
The BFS method explores all possible puzzle states level by level, checking the current state against the goal state. The algorithm uses a queue to store the states and their respective move paths.

    def BFS(self, maxnodes=MAX_NODES_LIMIT):
        queue = collections.deque([(self.curState, [])])
        nodes_created = 0
        while queue:
            if nodes_created >= maxnodes:
                print(f"Error: maxnodes limit ({maxnodes}) reached")
                return None
            current_state, path = queue.popleft()
            if current_state == self.GOAL_STATE:
                print(f"Goal reached with path: {path}")
                return path
            for direction in ["left", "right", "up", "down"]:
                if self.validMove(direction, current_state):
                    new_state = self.moveDirection(direction, state=current_state)
                    queue.append((new_state, path + [direction]))

DFS Search 
--------------------
DFS search has 2 functions -- one that uses an iterative stack and one that uses recursion. Func DFS Iterative stack popping the top of the stack and check with the goal state, going as deep as possible until no sucessors. Func DFSrecur uses recursion to bubble up the result after it either finds or do not find the goal state.

    def DFS(self, maxNodes=MAX_NODES_LIMIT, default_directions=DEFAULT_DIRECTIONS):
        ....
        stack = []
        visited = set()
        nodes_created = 0  
            
        stack.append((self.curState, []))
        visited.add(tuple(map(tuple, self.curState)))

    def DFS_recur(self, state=None, maxNodes=MAX_NODES_LIMIT, nodes_created=1, path=None, visited=None, default_directions=DEFAULT_DIRECTIONS):
        ....
        result = self.DFS_recur(state=new_state, maxNodes=maxNodes, nodes_created=nodes_created + 1, path=new_path, visited=visited, default_directions=default_directions)
        if result:  # If the goal was found, bubble up the result
            return result


A* Search
--------------------
Heuristic Selection

Based on the heuristic_name (h1 or h2), the corresponding heuristic function is selected.

Frontier (Priority Queue)

A Min-heap (frontier) is used to select the next state with the lowest f value, where  f = g + h. Pop the one with the lowest f value. The heap also contains the move_order (second value) in order to reorder when there is an 
f value tie.

    heapq.heappush(frontier, (f, 0, g, h, initial_state, []))  # Added tie-breaker with move order
    visited[initial_state_tuple] = g

If the new calculated f value of the new state is tie with another f value of a state in the min heap -> the second ordering will be move order. 
For example, new_f = 10 move_order = 0 (left), f = 10 (in the heap) move_order = 1 (right) => go to the second ordering (move_oreder) and process left first because 0 < 1.

Visited dictionary: Here, I use a dictionary to map the state to the cost to that state g(n). 
This is useful when we encounter a duplicated state, we can quickly map that state to the cost of that state, and compare the new cost of the duplicated state with the one currently in the dictionary. 
If the cost is lower -> we then replace that state in the dict of the old cost with the new cost. If the cost is higher -> we then skip the current iteration and move on to the next valid state.

    # if g > the cost of new_state_tuple means that there is a better path cost already leading to that tile. skip iteration
    if new_state_tuple in visited and new_g >= visited.get(new_state_tuple, float('inf')):
        continue

    # if new g of the new tuple is less than the g in the existing tuple (if not exist then inf) -> BETTER PATH leading to that tile therefore replace and push to heap
    if new_state_tuple not in visited or new_g < visited.get(new_state_tuple, float('inf')):
        visited[new_state_tuple] = new_g
        nodes_created += 1
        # if f tie break then order according to move order
        # ex: f = 10 move_order = 0 (left) f = 10 move_order = 1 (right) => process left first because 0 < 1 and min heap
        move_order = self.DEFAULT_DIRECTIONS.index(direction)
        heapq.heappush(frontier, (new_f, move_order, new_g, new_h, new_state, path + [direction]))
        # Duplicate state detected with bigger g not enter if

Termination Conditions: Goal Reached: If the current state matches the goal state, the path is returned. Max Nodes Limit: If the number of nodes created exceeds maxNodes, the search terminates without finding a solution.


Examples of Valid Commands
--------------------------
    setState 1 2 3 4 5 6 7 8 0
    move up
    move left
    scrambleState 10
    setSeed 20
    printState
    solve BFS
    solve DFS 
    solve BFS maxnodes=5000
