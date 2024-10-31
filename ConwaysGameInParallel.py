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
    def __init__(self, states, alphabet, transition_function, initial_state, final_states): #Automata instantiation method 
        self.states = states #The possible states of the automata (All are final for cellular automata representation)
        self.alphabet = alphabet #The input alphabet (0,1) in our case
        self.transition_function = transition_function #The cellular automata rule
        self.initial_state = initial_state #O state, only non-final state
        self.final_states = final_states #Equals the states in our case minus O
        self.current_state = initial_state #resets the automata after each input string
    
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
    } 

Conways_Game =  Automata(states1, alphabet, conways_game, initial_state, final_states1) #Declaration of Conway's game object.

def extract_active_region(grid): #Function to consider only sub-array where the subarray is defined as the smallest square grid containing all ones.
    grid = np.array(grid) #makes the grid an np.array
    rows, cols = grid.shape #Declares two variables assigned to the length and width of the whole array
    # Find the indices where "1"s are located
    ones_indices = np.argwhere(grid == 1) #Locates indices of all ones in the array
    
    if ones_indices.size == 0: #Checks if array is all zeros
        np.zeros_like(grid) #Creates a grid of zeros 
    return grid  #returns the grid of zeros as the active region
       
    # Find the bounds of the active region
    min_row, min_col = ones_indices.min(axis=0) #Bottom left corner of the active array based on the minimum one index
    max_row, max_col = ones_indices.max(axis=0) #Top right corner of the active array based on the maximum one index
    
    # Extract and return the bounding box containing all "1"s
    active_region = grid[min_row:max_row + 1, min_col:max_col + 1]
    return active_region

def expand_grid(grid, expansion_size = 1): #Function to expand grid by a given amount, defaulted at 1.
    grid = np.array(grid) #makes the array an np array
    original_shape = grid.shape #creates a pair of length width variables 
    
    new_shape = (original_shape[0] + 2 * expansion_size, original_shape[1] + 2 * expansion_size) #Adds the expansion size to the row size and the column size.
    expanded_grid = np.zeros(new_shape, dtype=grid.dtype) #Makes a grid of zeros in that shape

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

def submatrix2d(grid, submatrix_size): #A function to create the local neighborhood matrices of each cell in the grid
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

def kolmogorov_complexity(string: str) -> dict: #Function to estimate the Kolmogorov complexity
    binary_str = ''.join(map(str, string)) #turns the array [0,0,1,..] into a binary string 001...
    # Convert the string to byte
    data = binary_str.encode('utf-8')
    
    # Compress using zlib (gzip), bz2, and lzma
    zlib_compressed = zlib.compress(data)
    bz2_compressed = bz2.compress(data)
    lzma_compressed = lzma.compress(data)
    
    average_compressed_size = (len(zlib_compressed) + len(bz2_compressed) + len(lzma_compressed)) / 3
    # Return the size of the compressed data as an estimate of Kolmogorov complexity
    return average_compressed_size

def lempel_ziv_complexity(binary_sequence): #Function for finding Lemple-Ziv Complexity stolen from an article I will need to cite and figure out
    """Lempel-Ziv complexity for a binary sequence, in simple Python code."""
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
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

def Sequence_States(State, num_steps, rule, dimension): #Main function that performs the automata operation over a number of steps
    data_set = []  # Initialize an empty list to store the data
    entropy_list = []  # List to store entropy values over time
    growth = 0 #This is a counter to determine how much the automata naturally expands.
    phase = set((pos, tuple(config)) for pos, config in enumerate(State)) #This creates a set (no repeats) of tuples of the position of a cell and its dependency vector (phase space)
    initial_entropy = math.log2(len(phase)) #The log_2 of the phase space volume (or length) is the definition of Boltzmann Entropy (A physics concept)
    k = 1 / initial_entropy #Boltzmann entropy has a constant, we create a constant such that the entropy begins at 1
    print(k) #Print the constant (should be the log_2(size of the initial outer array))
    entropy_grow = 0 #cant remember dont think important
    for t in range(num_steps): #iterates over how long we want the automata to run
        print(f"k: {k}") #prints k at each time step (it was getting lost at the top otherwise)
        data = []  # Initialize dataset for this time step
        value_stateN = next_state(State, 5, 2, rule) # Get next state from FSM
        value_stateN1 = extract_active_region(value_stateN) #Extracts the region of ones from the system
        #print(f"value state for complexity: {value_stateN}")
        if dimension == 2: #if is only kept here as I stole this from the larger mother code
            value_stateConcat = list(itertools.chain.from_iterable(value_stateN1)) #Turns the whole 2d array into a binary string to be processed
            complexity_lz2d = lempel_ziv_complexity(value_stateConcat)   #determine complexity of active region
            complexity_k2d = kolmogorov_complexity(value_stateConcat)                  
#        print(f"Next state after FSM at t = {t+1}: {value_stateN}")
        print(f"value_stateN1_len: {len(value_stateN1)}") #Keeps track of the relative sizes of inner and outer grids (N1 inner, N outer)
        print(f"value_stateN_len: {len(value_stateN)}")
        if len(value_stateN1) == len(value_stateN): #Checks to see if the active region has reaches the same size as the outer grid
            value_stateN = expand_grid(value_stateN) #Expands the whole thing if so
            growth += 1 #Growth counter increases by 1
        print(f"value_state_len: {len(value_stateN)}")
        DependencyN = dependency_form(value_stateN, 3, "S", 2)  # Get dependency form of the value state
        Phase = dependency_form(value_stateN1, 3, "S", 2) #Determines the current phase space by the dependency form of current grid
        new_phase_entries = set((pos, tuple(config)) for pos, config in enumerate(Phase)) #notes any tuples that were produced in the step
        phase.update(new_phase_entries) #Adds new tuples to the set
        boltzmann_entropy = k * math.log2(len(phase)) #Calculates Boltzmann entropy of updates set
        print(boltzmann_entropy)    #prints it
        State = DependencyN #Updates the state to the dependency to be run again

        # For each element in the current state, create a data point
        for i in range(len(State)): #Tracks the count of all states for Shannon entropy calculation (I think this mightve been for the 1d automata)
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
        entropy = -np.sum(state_counts['probability'] * np.log2(state_counts['probability'] + 1e-10))  #Shannon entropy calculation
        entropy_grow += entropy
               
        entropy_list.append({ #Creates a list of all variables at every step.
            'time': t,
            'entropy_states': entropy,
            'complexity_serieslz2d': complexity_lz2d,
            'complexity_seriesk2d' : complexity_k2d,
            'boltzmann_entropy': boltzmann_entropy,
            'growth': growth
            
            })

    
    # Return the data set as a list
    return entropy_list

def random_inside_larger2d(outer_size, inner_size): #Function to create a square of randomly distributed ones and zeros inside a larger array of zeros
    grid_of_random = [[0]*inner_size]*inner_size #Initializes a grid of zeros set to the size inner_size*inner_size
    #print(value_state_stoch2d)
    for i in range(inner_size): #Over all indices in generates grid
        for j in range(inner_size):
            random_digit = random.randint(0, 1) #Generates a (normal?) distribution of ones and zeros
            grid_of_random[i][j] = random_digit #Updates that index with the one or zero
            
    outer_grid = np.zeros((outer_size,outer_size), dtype = int) #Creates an array of zeros of size outer_size*outer_size
    
    start_x = (outer_size - inner_size) // 2 #these find the indices of the edges of where the random array will go
    start_y = (outer_size - inner_size) // 2

    outer_grid[start_x:start_x + inner_size, start_y:start_y + inner_size] = grid_of_random #places the random swaure at those indices and returns

    return outer_grid

def run_simulation_on_initial(inner_size, num_steps, rule, dimension): #Runs an automata of a given inner size 
    outer_size = inner_size + 2 #For our purposes the outersize will be initialized at two larger than the smaller (gives some breathing room at initial steps)
    value_state_stoch2dp = random_inside_larger2d(outer_size, inner_size) #Creates an array of outer zeros and inner random
    State = dependency_form(value_state_stoch2dp, 3, "S", 2) #Generates teh dependency form of that grid
    return Sequence_States(State, num_steps, rule, dimension) #runs the automata

def run_parallel_simulations(inner_size_range, num_steps, rule, dimension, max_workers=4): #This stuff I hardly understand
    results = [] #Creates an array to hold our various results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # Use ThreadPoolExecutor
        futures = [
            executor.submit(run_simulation_on_initial, inner_size, num_steps, rule, dimension) #Runs the automata with max workers
            for inner_size in inner_size_range #runs over the range of inner sizes (want to find lower emergence bound as there seems to be a size at which things take off)
        ]
        for future in futures: #over all simulations in inner size range
            results.append(future.result())  #append the list to the results array
    return results 

all_dfs = [] #An array to hold our many lists 
inner_size_range = range(1, 33, 2) #Sets the range of our computation (This is what needs changed)
num_steps = 200 #Number of steps
rule = Conways_Game #The automata in use, I will add more when we run some batches on this
dimension = 2 #Dimension of the automata
inner_size = 25 #Sets initial size if we want to test simulation before running

results = run_parallel_simulations(inner_size_range, num_steps, rule, dimension) #run the parallel simulations of the automata


for i, result in enumerate(results): #For all the results in the results array
    df_result = pd.DataFrame(result) #make a data frame of each 
    df_result['initial_size'] = inner_size_range[i]  # Label with initial size
    all_dfs.append(df_result) #Add each data frame
    
df_all_entropy = pd.concat(all_dfs, ignore_index=True) #Combine them all into one chunky dataframe

file_path = r'C:\Users\benja\OneDrive\Desktop\ConwaysGameALLEntropy.xlsx' #This also needs changed based on your local machine name

# Save the DataFrame to an Excel file
df_all_entropy.to_excel(file_path, index=False)

print(f'File saved to: {file_path}') #lets you know it's done.
