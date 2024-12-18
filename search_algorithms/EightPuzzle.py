import random
import collections
import heapq


class Puzzle: 
    MAX_NODES_LIMIT = 1000
    DEFAULT_DIRECTIONS = ["left", "right", "up", "down"]
    GOAL_STATE = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]  
    INITIAL_STATE = [[1, 0, 2], [3, 4, 5], [6, 7, 8]]

    def __init__(self):
        self.curState = [row[:] for row in self.INITIAL_STATE]  
        self.goalState = [row[:] for row in self.GOAL_STATE]  
    
    def setSeed(self, seed_value):
        random.seed(seed_value)

    def setState(self, state):
        state_parts = state.split()
        
        if len(state_parts) != 9 or not all(part.isdigit() and 0 <= int(part) <= 8 for part in state_parts):
            print("Error: invalid puzzle state")
            return
        
        state_list = [int(part) for part in state_parts]  
        
        if sorted([x for x in state_list if x != 0]) != list(range(1, 9)):
            print("Error: invalid puzzle state")
            return
        
        matrix = self.list_to_matrix(state_list)

        self.curState = matrix
        print("New state set:")
        self.printState()

    def list_to_matrix(self, state_list):
        return [state_list[i:i+3] for i in range(0, 9, 3)]
        
    def printState(self, state=None):
        matrix = state if state else self.curState
        for row in matrix:
            # Replace 0 with a blank space when printing
            print(" ".join(str(x) if x != 0 else " " for x in row))

    def moveDirection(self, direction, state=None):
        matrix = [row[:] for row in (state if state else self.curState)]
        if not self.validMove(direction, matrix):
            print(f"Error: invalid move {direction}")
            return None

        blank_i, blank_j = self.findBlank(matrix)

        if direction == "left":
            matrix[blank_i][blank_j], matrix[blank_i][blank_j - 1] = matrix[blank_i][blank_j - 1], matrix[blank_i][blank_j]
        elif direction == "right":
            matrix[blank_i][blank_j], matrix[blank_i][blank_j + 1] = matrix[blank_i][blank_j + 1], matrix[blank_i][blank_j]
        elif direction == "up":
            matrix[blank_i][blank_j], matrix[blank_i - 1][blank_j] = matrix[blank_i - 1][blank_j], matrix[blank_i][blank_j]
        elif direction == "down":
            matrix[blank_i][blank_j], matrix[blank_i + 1][blank_j] = matrix[blank_i + 1][blank_j], matrix[blank_i][blank_j]

        if state is None:
            print(f"Moved {direction}:")
            self.curState = matrix
            self.printState()
        
        return matrix
    
    def findBlank(self, state=None):
        matrix = state if state else self.curState
        for i in range(3):
            for j in range(3):
                if matrix[i][j] == 0: 
                    return i, j
        return None

    def validMove(self, direction, state=None):
        matrix = state if state else self.curState
        # blank i row, blank j column
        blank_i, blank_j = self.findBlank(matrix)

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

    def scrambleState(self, n, default_directions=DEFAULT_DIRECTIONS):
        self.curState = [row[:] for row in self.GOAL_STATE]  
        for _ in range(n): 
            valid_directions = []
            for direction in default_directions:
                if self.validMove(direction):
                    valid_directions.append(direction)
            
            if valid_directions:
                move = random.choice(valid_directions)
                print(f"Scramble move: {move}")  
                self.moveDirection(move)


    def BFS(self, maxnodes=MAX_NODES_LIMIT, default_directions=DEFAULT_DIRECTIONS):
        queue = collections.deque([(self.curState, [])])
        visited = set()
        visited.add(tuple(map(tuple, self.curState)))
        nodes_created = 1

        while queue:
            if nodes_created >= maxnodes:
                print(f"Error: maxnodes limit ({maxnodes}) reached")
                return None, nodes_created
            
            current_state, path = queue.popleft()
            nodes_created += 1

            if current_state == self.GOAL_STATE:
                return path, nodes_created

            for direction in default_directions:
                if self.validMove(direction, current_state): 
                    new_state = self.moveDirection(direction, state=current_state)  
                    if new_state is None:
                        continue  # Invalid move, skip
                    new_state_tuple = tuple(map(tuple, new_state))
                    if new_state_tuple not in visited:  # Check if the state is already visited
                        visited.add(new_state_tuple)
                        new_path = path + [direction]
                        queue.append((new_state, new_path))

        return None, nodes_created

    def DFS(self, maxNodes=MAX_NODES_LIMIT, default_directions=DEFAULT_DIRECTIONS):
        stack = []
        visited = set()
        nodes_created = 0  
        
        stack.append((self.curState, []))
        visited.add(tuple(map(tuple, self.curState)))
        
        while stack and nodes_created < maxNodes:
            current_state, path = stack.pop()
            nodes_created += 1

            if current_state == self.GOAL_STATE:
                return path, nodes_created

            for direction in default_directions:
                if self.validMove(direction, current_state):
                    new_state = self.moveDirection(direction, state=current_state)  
                    if new_state is None:
                        continue  # Invalid move, skip
                    new_state_tuple = tuple(map(tuple, new_state))
                    if new_state_tuple not in visited: 
                        visited.add(new_state_tuple)
                        new_path = path + [direction]  
                        stack.append((new_state, new_path))

        return None, nodes_created

    def DFS_recur(self, state=None, maxNodes=MAX_NODES_LIMIT, nodes_created=1, path=None, visited=None, default_directions=DEFAULT_DIRECTIONS):
        if visited is None:
            visited = set()
        if state is None:
            state = self.curState
        if path is None:
            path = []
        
        state_tuple = tuple(map(tuple, state)) 
        if state_tuple in visited:
            return None, nodes_created  

        if state == self.GOAL_STATE:
            return path, nodes_created  

        if nodes_created >= maxNodes:
            return None, nodes_created  

        visited.add(state_tuple)  

        for direction in default_directions:
            if self.validMove(direction, state):
                new_state = self.moveDirection(direction, state=state)
                if new_state is None:
                    continue  # Invalid move, skip
                new_state_tuple = tuple(map(tuple, new_state)) 

                if new_state_tuple not in visited:  
                    new_path = path + [direction]
                    
                    result, nodes_created = self.DFS_recur(state=new_state, maxNodes=maxNodes, nodes_created=nodes_created + 1, path=new_path, visited=visited, default_directions=default_directions)
                    if result:  # If the goal was found, bubble up the result
                        return result, nodes_created

        return None, nodes_created

    def heuristic_h1(self, state=None):
        state = state if state else self.curState
        misplaced = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0 and state[i][j] != self.GOAL_STATE[i][j]:
                    misplaced += 1
        return misplaced


    def heuristic_h2(self, state=None):
        state = state if state else self.curState
        distance = 0
        for i in range(3):
            for j in range(3):
                tile = state[i][j]
                if tile != 0:
                    goal_i, goal_j = self.findPosition_h2(tile)
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance

    def findPosition_h2(self, tile):
        for i in range(3):
            for j in range(3):
                if self.GOAL_STATE[i][j] == tile:
                    return i, j
        return None

    def A_star(self, heuristic_name, maxNodes=MAX_NODES_LIMIT):
        heuristic_func = None
        if heuristic_name == 'h1':
            heuristic_func = self.heuristic_h1
        elif heuristic_name == 'h2':
            heuristic_func = self.heuristic_h2
        else:
            print(f"Unknown heuristic: {heuristic_name}")
            return None, 0

        frontier = []
        heapq.heapify(frontier)
        visited = {}

        initial_state = self.curState
        initial_state_tuple = tuple(map(tuple, initial_state))
        g = 0
        h = heuristic_func(initial_state)
        f = g + h
        nodes_created = 1

        heapq.heappush(frontier, (f, 0, g, h, initial_state, []))  # Added tie-breaker with move order
        visited[initial_state_tuple] = g

        while frontier:
            if nodes_created >= maxNodes:
                print(f"Error: maxnodes limit ({maxNodes}) reached")
                return None, nodes_created

            f, _, g, h, current_state, path = heapq.heappop(frontier)

            if current_state == self.GOAL_STATE:
                return path, nodes_created

            for direction in self.DEFAULT_DIRECTIONS:
                if self.validMove(direction, current_state):
                    new_state = self.moveDirection(direction, state=current_state)
                    if new_state is None:
                        continue  # Invalid move, skip
                    new_state_tuple = tuple(map(tuple, new_state))
                    new_g = g + 1
                    new_h = heuristic_func(new_state)
                    new_f = new_g + new_h

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

        return None, nodes_created

    @staticmethod
    def effective_branching_factor(d, N, tolerance=1e-6, max_iterations=1000):
        if d < 0:
            raise ValueError("Depth d must be non-negative.")
        if N < 0:
            raise ValueError("Total number of nodes N must be non-negative.")
        if d == 0:
            return 0.0 if N == 0 else float('inf')
        if d == 1:
            return float(N)  

        # Formula to calculate b*
        def total_nodes(b):
            try:
                return sum(b**i for i in range(d + 1))
            except OverflowError:
                return float('inf') 

        left = 1e-6  
        right = max(2.0, N) if N > 0 else 1.0  

        iteration = 0
        target = N 
        # Binary Search 
        while iteration < max_iterations:
            mid = (left + right) / 2
            estimate = total_nodes(mid)  

            if abs(estimate - target) < tolerance:  
                return mid
            elif estimate < target:
                left = mid  
            else:
                right = mid  

            iteration += 1

        return (left + right) / 2


    def calculate_efb(self, depth, total_nodes):
        return Puzzle.effective_branching_factor(depth, total_nodes)

    def generate_test_cases(self, scramble_moves_list):
        test_cases = []
        
        for scramble_moves in scramble_moves_list:
            difficulty = f'ScrambleMoves={scramble_moves}'
            
            print(f"\nGenerating {difficulty} test case by scrambling {scramble_moves} moves:")
            self.setSeed(10)
            self.scrambleState(scramble_moves)
            initial_state = [row[:] for row in self.curState]
            test_cases.append((difficulty, initial_state))
            self.curState = [row[:] for row in self.GOAL_STATE]  # Reset to goal for next scramble

        return test_cases

    def print_solution(self, path, nodes_generated):
        if path:
            print(f"Goal reached with path: {path}")
            print(f"Nodes created during search: {nodes_generated}")
            print(f"Solution length: {len(path)}")
            print("Move sequence:")
            for move in path:
                print(f"move {move}")


def compare_search_algorithms(puzzle, test_cases, algorithms):
    efb_results = {}
    nodes_results = {}
    depth_results = {}

    for difficulty, initial_state in test_cases:
        scramble = difficulty
        efb_results[scramble] = {}
        nodes_results[scramble] = {}
        depth_results[scramble] = {}
        for algo in algorithms:
            print(f"\nRunning {algo} on {difficulty} test case:")
            # Set the puzzle to the initial state
            puzzle.setState(' '.join(str(cell) for row in initial_state for cell in row))
            
            # Run the selected algorithm
            if algo == 'DFS':
                path, nodes_generated = puzzle.DFS(300000)
            elif algo == 'BFS':
                path, nodes_generated = puzzle.BFS(300000)
            elif algo == 'A*_h1':
                path, nodes_generated = puzzle.A_star('h1')
            elif algo == 'A*_h2':
                path, nodes_generated = puzzle.A_star('h2')
            else:
                print(f"Unknown algorithm: {algo}")
                continue

            depth = len(path) if path else 'No Solution'
            
            if path:
                b_star = puzzle.calculate_efb(depth, nodes_generated)
                b_star = round(b_star, 4)
            else:
                b_star = 'N/A'

            efb_results[scramble][algo] = b_star
            nodes_results[scramble][algo] = nodes_generated
            depth_results[scramble][algo] = depth

    return efb_results, nodes_results, depth_results

def display_results_table(efb_results, nodes_results, depth_results):
    def display_table(title, headers, data_dict):
        print(f"\n{title}\n")
        header_row = " | ".join(f"{header:<16}" for header in headers)  
        print(f"| {header_row} |")
        # Create separator
        separator = "| " + " | ".join("-" * 16 for _ in headers) + " |"
        print(separator)
        
        # Populate rows
        for scramble, metrics in data_dict.items():
            row = f"| {scramble:<16} | " + " | ".join(f"{str(metrics.get(algo, 'N/A')):<16}" for algo in headers[1:]) + " |"
            print(row)
    
    algorithms = ['BFS', 'DFS', 'A*_h1', 'A*_h2']
    
    # Efective Branching Factor Table
    display_table(
        "Effective Branching Factor (b*)",
        ["ScrambleMoves"] + algorithms,
        efb_results
    )
    
    # Total Nodes Generated Table
    display_table(
        "Total Nodes Generated",
        ["ScrambleMoves"] + algorithms,
        nodes_results
    )
    
    # Depth Table
    display_table(
        "Depth",
        ["ScrambleMoves"] + algorithms,
        depth_results
    )


def run_efc(parameters):
    depth = None
    N = None
    
    # Parse 
    for param in parameters:
        if param.startswith("d="):
            try:
                depth = int(param.split("=")[1])
            except ValueError:
                print(f"Error: Invalid depth value in parameter '{param}'. Depth must be an integer.")
                return
        elif param.startswith("n="):
            try:
                N = int(param.split("=")[1])
            except ValueError:
                print(f"Error: Invalid node count value in parameter '{param}'. Node count must be an integer.")
                return
        else:
            print(f"Error: Unknown parameter '{param}'. Expected format 'd=<depth> n=<total_nodes>'.")
            return
    
    if depth is None or N is None:
        print("Error: Both depth (d=) and node count (n=) parameters must be provided.")
        return
    
    if depth < 0:
        print("Error: Depth (d) must be non-negative.")
        return
    if N < 0:
        print("Error: Total number of nodes (n) must be non-negative.")
        return
    
    puzzle = Puzzle()
    b_star = puzzle.calculate_efb(depth, N)
    
    b_star_rounded = round(b_star, 3)
    
    print(f"Result: efc d={depth} n={N} b*={b_star_rounded}")

def generate_table(scramble_moves):
    puzzle = Puzzle()
    test_cases = puzzle.generate_test_cases(scramble_moves)
    algorithms = ['DFS', 'BFS', 'A*_h1', 'A*_h2']
    efb_results, nodes_results, depth_results = compare_search_algorithms(puzzle, test_cases, algorithms)
    display_results_table(efb_results, nodes_results, depth_results)

def cmd(command_str, puzzle, line_num=None):
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
        elif command == "heuristic":
            heuristic_name = parts[1]
            if heuristic_name == 'h1':
                value = puzzle.heuristic_h1()
                print(f"Misplaced tiles: {value}")
            elif heuristic_name == 'h2':
                value = puzzle.heuristic_h2()
                print(f"Total Manhattan distance: {value}")
            else:
                print(f"Unknown heuristic: {heuristic_name}")
        elif command == "solve":
            if len(parts) < 2:
                print(f"Error: invalid command: {command_str} on line {line_num}")
                return
            strategy = parts[1]
            nodes = 1000 
            
            if len(parts) > 2 and parts[2].startswith("maxnodes="):
                try:
                    nodes = int(parts[2].split('=')[1])  
                except ValueError:
                    print(f"Error: invalid maxnodes value in command: {command_str} on line {line_num}")
                    return
                
            if strategy == "BFS":
                path, nodes_generated = puzzle.BFS(nodes)
                if path:
                    puzzle.print_solution(path, nodes_generated)
                else:
                    print(f"No solution found within {nodes} nodes.")
            elif strategy == "DFS":
                path, nodes_generated = puzzle.DFS(nodes)
                if path:
                    puzzle.print_solution(path, nodes_generated)
                else:
                    print(f"No solution found within {nodes} nodes.")
            elif strategy == "A*":
                if len(parts) < 3:
                    print(f"Error: missing heuristic for A* in command: {command_str} on line {line_num}")
                    return
                heuristic_name = parts[2]
                nodes = 1000
                if len(parts) > 3 and parts[3].startswith("maxnodes="):
                    try:
                        nodes = int(parts[3].split('=')[1])
                    except ValueError:
                        print(f"Error: invalid maxnodes value in command: {command_str} on line {line_num}")
                        return
                path, nodes_generated = puzzle.A_star(heuristic_name, maxNodes=nodes)
                if path:
                    puzzle.print_solution(path, nodes_generated)
                else:
                    print(f"No solution found within {nodes} nodes.")
            else:
                print(f"Error: unknown strategy '{strategy}' in command: {command_str}")
        elif command == "efc":
            parameters = parts[1:]
            run_efc(parameters)
        elif command == "generateTable":
            if len(parts) < 2:
                print(f"Error: 'generateTable' requires at least one scramble move argument.")
                return
            try:
                scramble_moves = [int(move) for move in parts[1:]]
                generate_table(scramble_moves)
            except ValueError:
                print(f"Error: Scramble moves must be integers. Received: {parts[1:]}")
        else:
            print(f"Error: invalid command: {command_str} on line {line_num}")
    except (IndexError, ValueError) as e:
        print(f"Error: invalid command: {command_str} on line {line_num}")

def cmdfile(filename):
    puzzle = Puzzle()  
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):  
            line = line.strip()
            if line.startswith("#") or line.startswith("//"):
                print(line)  
            elif line:
                print(line)  
            cmd(line, puzzle, line_num) 
    

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