import numpy as np  #Imports
import pandas as pd
import random
import zlib
import bz2
import lzma
import itertools
import pdb
import math
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import traceback

#pdb.set_trace()
class Automata: #Define the automata class
    def __init__(self, states, alphabet, transition_function, initial_state, final_states):
        self.states = states  #Declare object attributes
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.initial_state = initial_state
        self.final_states = final_states
        self.current_state = initial_state
    
    def transition(self, symbol):#Define transition method acting on self, and the input symbol
        if (self.current_state, symbol) in self.transition_function:   #If the symbol has a valid state transition
            self.current_state = self.transition_function[(self.current_state, symbol)] #Do the transition function on the state.
    
    def is_accepting(self): #def function for if the string accepts into the machine (ends on a final state)
        return self.current_state in self.final_states #Return the state after string input

    def reset(self): #Resets the machine to unitial state
        self.current_state = self.initial_state
        
    
        
#for rule 30
states1 = {"D01","A11","A01","D11","A02","D12", "A13", "D13", "D14", "N1", "N0", "A03", "D04", "D05"}  #Based on the diagram of the rule
alphabet = {0,1} #Input is always a 0 or 1
final_states1 = states1  #All states are final states
initial_state = "O" #Define O as the initial state

conways_game = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"D01",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("D12",1):"A11",
    ("D12",0):"D12",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("A12",1):"D13",
    ("A12",0):"A12", 
    ("D13",1):"D13",
    ("D13",0):"D13",
    ("D01",1):"D02",
    ("D01",0):"D01",
    ("D02",1):"D03",
    ("D02",0):"D02",
    ("D03",1):"A01",
    ("D03",0):"D03",
    ("A01",1):"D13",
    ("A01",0):"A01"
    } 

Conways_Game =  Automata(states1, alphabet, conways_game, initial_state, final_states1)

def extract_active_region(grid):
    grid = np.array(grid)
    rows, cols = grid.shape
    # Find the indices where "1"s are located
    ones_indices = np.argwhere(grid == 1)
    
    if ones_indices.size == 0:
        return np.zeros((3, 3), dtype=int)  # Return a minimal inactive region
       
    # Find the bounds of the active region
    min_row, min_col = ones_indices.min(axis=0)
    max_row, max_col = ones_indices.max(axis=0)
    
    # Extract and return the bounding box containing all "1"s
    active_region = grid[min_row:max_row + 1, min_col:max_col + 1]
    return active_region

def expand_grid(grid, expansion_size = 1):
    grid = np.array(grid)
    original_shape = grid.shape
    
    new_shape = (original_shape[0] + 2 * expansion_size, original_shape[1] + 2 * expansion_size)
    expanded_grid = np.zeros(new_shape, dtype=grid.dtype)

    # Place the original grid in the center of the new expanded grid
    expanded_grid[expansion_size:expansion_size + original_shape[0], 
    expansion_size:expansion_size + original_shape[1]] = grid

    return expanded_grid

def next_state(System_state, grid_size, dimension, rule): #Define next state function with state of system, size of total grid, and automata dimension
    if dimension == 1: #For 1d automata case
        value_stateN = np.zeros(grid_size, dtype=int).tolist() #Initialize a value_state vector of 0s
        #print(value_stateN)
    
        for i, vector in enumerate(System_state): #For each vector in the system state (dependency form)
            for symbol in vector: #for each symbol, 0 or 1 in the vector
                rule.transition(symbol) #Permorm the transition function on that 1 or 0
                # print(f"Current state: {Rule_32.current_state}")
            if rule.current_state[0] == "A": #If the state is an A type, the value vector at that index is 1
                    value_stateN[i] = 1
            elif rule.current_state[0] == "D": #If a D type, then the vector at this index is 0
                value_stateN[i] = 0
            rule.reset() #Reset the automata for next vector
                        #print(f"System State t+1 = {value_stateN}")
        return value_stateN #Return the value state (What the grid actually looks like)
    
    if dimension == 2: #2d case
        value_stateN = np.zeros((grid_size, grid_size), dtype=int).tolist() #Creates a matrix of zeros instead of vector
        #print(f"value_staten1 :{value_stateN}")
        #print(f"System_state2 : {System_state}")
        
        size = int(len(System_state) ** 0.5)  # Calculate the size of the square matrix
        value_stateN = np.zeros((size, size), dtype=int).tolist()  # Creates a square matrix of zeros
        
        for i in range(len(System_state)):  # Iterate over the flattened System_state
            row = System_state[i]  # Get the current row from System_state
            #print(f"Processing row {i}: {row}")
            
            # Process the current 3x3 neighborhood
            for value in row:  # Iterate over the values in the current row
                rule.transition(value)  # Directly transition based on value
        
            # Update the matrix after processing the entire neighborhood
            if rule.current_state[0] == "A":
                matrix_i = i // size  # Determine row index in the square matrix
                matrix_j = i % size   # Determine column index in the square matrix
                value_stateN[matrix_i][matrix_j] = 1
            elif rule.current_state[0] == "D":
                matrix_i = i // size  # Determine row index in the square matrix
                matrix_j = i % size   # Determine column index in the square matrix
                value_stateN[matrix_i][matrix_j] = 0 

            rule.reset()  # Reset the FSM for the next iteration

        return value_stateN  # Return the value state (What the grid actually looks like)

    return value_stateN  # Return the value state (What the grid actually looks like)

def submatrix2d(grid, submatrix_size):
    submatrices = []
    rows = len(grid)
    cols = len(grid[0])
    sub_rows, sub_cols = submatrix_size
    
    # Iterate over the grid and extract submatrices
    for i in range(rows - sub_rows + 1):
        for j in range(cols - sub_cols + 1):
            # Extract the submatrix and convert it to a list of lists
            submatrix = [row[j:j + sub_cols] for row in grid[i:i + sub_rows]]
            submatrices.append(submatrix)

    return submatrices

def dependency_form(value_state, kernel_width, dependency, dimension): #Define the dependency form function
    if dimension == 2:  # 2D case
        Kernel = np.ones((kernel_width, kernel_width), dtype=int)  # Define a kernel of ones
        #print(f"Kernel: {Kernel}")

        # Pad the grid (value_state) with zeros
        value_state_pad = np.pad(value_state, ((1, 1), (1, 1)), 'constant', constant_values=(0))
        #print(f"value_state_pad: {value_state_pad}")  # Padded grid

        System_StateN = np.zeros((len(value_state), len(value_state[0]), 9), dtype=int)  # Store the resulting 9-digit vectors

        System_StateNP = submatrix2d(value_state_pad, (kernel_width, kernel_width))  # Extract 2D submatrices
        #print(f"System_StateNP: {System_StateNP}")

        if dependency == "P":  # Position dependent
            for i in range(len(System_StateN)):
                for j in range(len(System_StateN[0])):
                    submatrix = np.array(System_StateNP[i * len(System_StateN[0]) + j]).flatten()
                    result = np.multiply(Kernel.flatten(), submatrix)  # Apply kernel
                    System_StateN[i][j] = result  # Store the 9-digit vector

        elif dependency == "S":  # State dependent
            System_StateN = []
            for i in range(len(System_StateNP)):
                submatrix = System_StateNP[i]  # Extract the current 3x3 list
                center = submatrix[1][1]  # Get the center value
                # Flatten the submatrix
                flattened = [item for row in submatrix for item in row]
                # Create the reordered vector with the center value first
                reordered_vector = [center] + flattened[:4] + flattened[5:]
            
                # Append the reordered vector to the temporary list
                System_StateN.append(reordered_vector)      
             
    return System_StateN

def partitions_by_all2d():
    partitions = {}
    for i in range(512):
        binary_str = format(i, '09b')  # Create a binary string of length 9
        partition = [int(bit) for bit in binary_str]  # Convert each bit to an integer
        partitions[i] = partition  # Assign the binary vector to the dictionary with key i
    return partitions

def kolmogorov_complexity(string: str) -> dict:
    binary_str = ''.join(map(str, string))
    # Convert the string to bytes
    data = binary_str.encode('utf-8')
    
    # Compress using zlib (gzip), bz2, and lzma
    zlib_compressed = zlib.compress(data)
    bz2_compressed = bz2.compress(data)
    lzma_compressed = lzma.compress(data)
    
    average_compressed_size = (len(zlib_compressed) + len(bz2_compressed) + len(lzma_compressed)) / 3
    # Return the size of the compressed data as an estimate of Kolmogorov complexity
    return average_compressed_size

def lempel_ziv_complexity(binary_sequence):
    """Lempel-Ziv complexity for a binary sequence, in simple Python code."""
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence) - 1
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
    return complexity

def Sequence_States(State, max_steps, rule, dimension, window_size = 75  , tol = 0.5, oscil_tol = 8): 
    data_set = []  # Initialize an empty list to store the data
    entropy_list = []  # List to store entropy values over time
    growth = 0
    phase = set((pos, tuple(config)) for pos, config in enumerate(State))
    initial_entropy = math.log2(len(phase))
    k = 1 / initial_entropy
    print(k)
    entropy_grow = 0
    complexities = []
    for t in range(max_steps): 
        #print(f"k: {k}")
        data = []  # Initialize dataset for this time step
        value_stateN = next_state(State, 5, 2, rule) # Get next state from FSM
        value_stateN1 = extract_active_region(value_stateN)
        #print(f"value state for complexity: {value_stateN}")
        if dimension == 2:
            value_stateConcat = list(itertools.chain.from_iterable(value_stateN1))
            complexity_lz2d = lempel_ziv_complexity(value_stateConcat)   
            complexity_k2d = kolmogorov_complexity(value_stateConcat)                   
        #print(f" {t+1}")
        #print(f"value_stateN1_len: {len(value_stateN1)}")
        #print(f"value_stateN_len: {len(value_stateN)}")
        if len(value_stateN1) == len(value_stateN):
            value_stateN = expand_grid(value_stateN)
            growth += 1
       # print(f"value_state_len: {len(value_stateN)}")
        DependencyN = dependency_form(value_stateN, 3, "S", 2)  # Get dependency form of the value state
        Phase = dependency_form(value_stateN1, 3, "S", 2)
        new_phase_entries = set((pos, tuple(config)) for pos, config in enumerate(Phase))
        phase.update(new_phase_entries)
        boltzmann_entropy = k * math.log2(len(phase))
        #print(boltzmann_entropy)      
        State = DependencyN
        
        complexities.append(complexity_lz2d)
        if len(complexities) > window_size:
            complexities.pop(0)
        
        oscil_tol = math.floor(oscil_tol / 10)*oscil_tol
        if oscil_tol == 0:
             oscil_tol = 10
        if len(complexities) == window_size:
            if np.allclose(complexities, complexities[0], atol = oscil_tol):
                print(f"system has stabilized at time:{t}")
                break 
            
        #print(complexity_lz2d)
        # For each element in the current state, create a data point
        for i in range(len(State)): 
            if i < len(State): 
                data.append({
                    'time': t,
                    'x': State[i][0],
                    'y': State[i][1],
                    'z': State[i][2],
                    'point_index': i
                })
        
        # Append data to data_set for all time steps
        data_set.extend(data)

        # Convert the dataset to a pandas DataFrame
        data_frame = pd.DataFrame(data_set)

        # Group the data points by their x, y, z coordinates and count occurrences
        state_counts = data_frame.groupby(['x', 'y', 'z']).size().reset_index(name='count')
        
        # Calculate the total number of states and probabilities
        total_states = state_counts['count'].sum()
        state_counts['probability'] = state_counts['count'] / total_states
        
        # Calculate entropy for this time step
        entropy = -np.sum(state_counts['probability'] * np.log2(state_counts['probability'] + 1e-10))  
        entropy_grow += entropy
               
        entropy_list.append({
            'time': t,
            'entropy_states': entropy,
            'complexity_serieslz2d': complexity_lz2d,
            'complexity_seriesk2d' : complexity_k2d,
            'boltzmann_entropy': boltzmann_entropy,
            'growth': growth
            
            })

    
    # Return the data set as a DataFrame and the entropy evolution over time
    return entropy_list

def random_inside_larger2d_proj(inner_size):
    outer_size = inner_size + 2
    grid_of_random = [[0]*inner_size]*inner_size
    #print(value_state_stoch2d)
    for i in range(inner_size):
        for j in range(inner_size):
            random_digit = random.randint(0, 1)
            grid_of_random[i][j] = random_digit
            
    outer_grid = np.zeros((outer_size,outer_size), dtype = int)
    
    start_x = (outer_size - inner_size) // 2
    start_y = (outer_size - inner_size) // 2

    outer_grid[start_x:start_x + inner_size, start_y:start_y + inner_size] = grid_of_random

    return outer_grid

def random_inside_larger2d_stoch(inner_size):
    outer_size = inner_size + 2
    grid_of_random = [[random.randint(0, 1) for _ in range(inner_size)] for _ in range(inner_size)]
           
    outer_grid = np.zeros((outer_size,outer_size), dtype = int)
    
    start_x = (outer_size - inner_size) // 2
    start_y = (outer_size - inner_size) // 2

    outer_grid[start_x:start_x + inner_size, start_y:start_y + inner_size] = grid_of_random

    return outer_grid

def run_simulation_on_initial(inner_size, num_steps, rule, dimension, seed_state_func):
    outer_size = inner_size + 2
    value_state_stoch2dp = seed_state_func(outer_size, inner_size)
    State = dependency_form(value_state_stoch2dp, 3, "S", 2)
    return Sequence_States(State, num_steps, rule, dimension)

def run_parallel_simulations(inner_size_range, num_steps, rule, dimension, max_workers=5 , repeat = 50):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Use ThreadPoolExecutor
        futures = [
            executor.submit(run_simulation_on_initial, inner_size, num_steps, rule, dimension, seed_state_func)
            for inner_size in inner_size_range
            for _ in range(repeat)
        ]
        for future in futures:
            results.append(future.result())
    return results

all_dfs = []
inner_size_range = [17]
num_steps = 2000
rule = Conways_Game
dimension = 2
inner_size = 25
repeats = range(50)
seed_state_func = random_inside_larger2d_proj(inner_size)
#seed_state_func = random_inside_larger2d_stoch(inner_size)

results = run_parallel_simulations(inner_size_range, num_steps, rule, dimension)


for i, result in enumerate(results):
    df_result = pd.DataFrame(result)
    df_result['initial_size'] = repeats[i]  # Label with initial size
    all_dfs.append(df_result)
    
df_all_entropy = pd.concat(all_dfs, ignore_index=True)

file_path = r'C:\Users\benja\OneDrive\Desktop\ConwaysGametillstable15.xlsx' 

# Save the DataFrame to an Excel file
df_all_entropy.to_excel(file_path, index=False)

print(f'File saved to: {file_path}')
