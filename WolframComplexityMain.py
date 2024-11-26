import numpy as np  #Imports
import pandas as pd
import random
import zlib
import bz2
import lzma
import itertools
import time
import pdb
import math

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


transition_function101 = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"D01",
    ("D11",1):"A11",
    ("D11",0):"D11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A11",1):"D12",
    ("A11",0):"A11",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function30 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function54 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D11",
    ("A11",0):"A13",
    ("A13",1):"A12",
    ("A13",0):"A13",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("A01",1):"D02",
    ("A01",0):"A01",  
    }

transition_function14 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D13",
    ("A11",0):"D11",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("D13",1):"D14",
    ("D13",0):"D13",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function110 = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"D01",
    ("D11",1):"A11",
    ("D11",0):"D11",
    ("A11",1):"D12",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function126 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("A12",1):"D12",
    ("A12",0):"A12",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function150 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D11",1):"A12",
    ("D11",0):"D11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"D02",
    ("A01",0):"A01",  
    }

transition_function90 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"N01",
    ("N11",1):"A12",
    ("N11",0):"A11",
    ("A12",1):"D12",
    ("A12",0):"A12",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("N01",1):"D02",
    ("N01",0):"D01",
    ("D02",1):"A02",
    ("D02",0):"D02",
    ("D01",1):"A01",
    ("D01",0):"D01",
    }

transition_function22 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"D01",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"D02",
    ("A01",0):"A01",  
    }

transition_function60 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"N01",
    ("N11",1):"D11",
    ("N11",0):"A11",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("N01",1):"A01",
    ("N01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",
    ("D01",1):"D01",
    ("D01",0):"D01",
    }

transition_function222 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"D01",
    ("N11",1):"A12",
    ("N11",0):"A11",
    ("A12",1):"A13",
    ("A12",0):"A12",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",
    }

transition_function188 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"N01",
    ("N11",1):"D11",
    ("N11",0):"A11",
    ("D11",1):"A13",
    ("D11",0):"D11",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("N01",1):"A02",
    ("N01",0):"D01",
    ("A02",1):"A03",
    ("A02",0):"A02",
    ("D01",1):"A01",
    ("D01",0):"D01",
    }

transition_function62 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"D01",
    ("N11",1):"D11",
    ("N11",0):"A11",
    ("D11",1):"D12",
    ("D11",0):"D11",
    ("A11",1):"A12",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"A02",
    ("A01",0):"A01",
    }

transition_function86 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"D01",
    ("N11",1):"A12",
    ("N11",0):"A11",
    ("A12",1):"D12",
    ("A12",0):"A12",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"D02",
    ("A01",0):"A01",
    }

transition_function147 = { #Transition function, (state, input symbol):next state
    ("O",1):"A11",
    ("O",0):"N01",
    ("A11",1):"D11",
    ("A11",0):"A11",
    ("D11",1):"A12",
    ("D11",0):"D11",
    ("N01",1):"D01",
    ("N01",0):"A01",
    ("D01",1):"D02",
    ("D01",0):"D02",
    ("A01",1):"A02",
    ("A01",0):"A01",  
    }

transition_function4 = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"N01",
    ("D11",1):"D11",
    ("D11",0):"D12",
    ("D12",1):"D13",
    ("D12",0):"D12",
    ("N01",1):"A01",
    ("N01",0):"D01",
    ("A01",1):"D03",
    ("A01",0):"A01",
    ("D01",1):"D02",
    ("D01",0):"D01",  
    }

transition_function102 = { #Transition function, (state, input symbol):next state
    ("O",1):"D11",
    ("O",0):"D01",
    ("D11",1):"A11",
    ("D11",0):"D11",
    ("A11",1):"D12",
    ("A11",0):"A11",
    ("D01",1):"A01",
    ("D01",0):"D01",
    ("A01",1):"D02",
    ("A01",0):"A01",  
    } 

transition_function45 = { #Transition function, (state, input symbol):next state
    ("O",1):"N11",
    ("O",0):"N01",
    ("N11",1):"D12",
    ("N11",0):"D11",
    ("D12",1):"D13",
    ("D12",0):"D12",
    ("D11",1):"A11",
    ("D11",0):"D11",
    ("N01",1):"A02",
    ("N01",0):"A01",
    ("A02",1):"A03",
    ("A02",0):"A02",
    ("A01",1):"D01",
    ("A01",0):"A01",
    }

#2d automata
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

High_life = { #Transition function, (state, input symbol):next state
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
    ("A01",1):"D04",
    ("A01",0):"A01",
    ("D04",1):"D05",
    ("D04",0):"D04",
    ("D05",1):"A02",
    ("D05",0):"D05",
    ("A02",1):"D12",
    ("A02",0):"A02"
    } 

Rule_30 = Automata(states1, alphabet, transition_function30, initial_state, final_states1) #Define rule32 object with the given attributes
Rule_101 = Automata(states1, alphabet, transition_function101, initial_state, final_states1)
Rule_54 = Automata(states1, alphabet, transition_function54, initial_state, final_states1)
Rule_14 = Automata(states1, alphabet, transition_function14, initial_state, final_states1)
Rule_110 = Automata(states1, alphabet, transition_function110, initial_state, final_states1)
Rule_126 = Automata(states1, alphabet, transition_function126, initial_state, final_states1)
Rule_150 = Automata(states1, alphabet, transition_function150, initial_state, final_states1)
Rule_90 = Automata(states1, alphabet, transition_function90, initial_state, final_states1)
Rule_22 = Automata(states1, alphabet, transition_function22, initial_state, final_states1)
Rule_60 = Automata(states1, alphabet, transition_function60, initial_state, final_states1)
Rule_222 = Automata(states1, alphabet, transition_function222, initial_state, final_states1)
Rule_188 = Automata(states1, alphabet, transition_function188, initial_state, final_states1)
Rule_62 = Automata(states1, alphabet, transition_function62, initial_state, final_states1)
Rule_86 = Automata(states1, alphabet, transition_function86, initial_state, final_states1)
Rule_147 = Automata(states1, alphabet, transition_function147, initial_state, final_states1)
Rule_4 =  Automata(states1, alphabet, transition_function4, initial_state, final_states1)
Rule_102 =  Automata(states1, alphabet, transition_function102, initial_state, final_states1)
Rule_45 =  Automata(states1, alphabet, transition_function45, initial_state, final_states1)
#2d rules
Conways_Game =  Automata(states1, alphabet, conways_game, initial_state, final_states1)
High_life =   Automata(states1, alphabet, High_life, initial_state, final_states1)



System_state0 = [(0, 0, 0), (1, 0, 1),(0, 1, 1),(1, 1, 0),(1, 0, 0)] #Example initial states for the system in 1d
System_state1 = [(0, 1, 0), (1, 1, 1),(0, 1, 0),(0, 0, 0),(1, 1, 0)]
System_state2 = [(0, 1, 1), (0, 0, 1),(1, 1, 1),(1, 0, 0),(0, 0, 1)]

#System_state2d = [[(0,1,0),(1,0,1),(0,1,0)], [(0,1,0),(1,0,1),(0,1,0)], [(0,1,0),(1,0,1),(0,1,0)]] did this wrong

def mean_distance(state, initial_rows, initial_cols):
    initial_max_distance = initial_rows // 2 + initial_cols // 2
    
    rows = len(state)
    cols = len(state[0])
    middle_row = rows // 2
    middle_col = cols // 2
    
    distance_count = 0
    one_count = 0
    
    max_distance = initial_max_distance
    
    for i in range(rows):
        for j in range(cols):
            if state[i][j] == 1:
                # Corrected calculation of distance
                distance = abs(middle_row - i) + abs(middle_col - j)
                distance_count += distance
                one_count += 1
    
    # Avoid division by zero in case there are no 1s
    distance_avg = distance_count / one_count if one_count > 0 else 0
    normalized_distance = distance_avg / (max_distance)

    return normalized_distance * np.log(len(state)*len(state[0]))

def normalized_entropy(grid):
    # Calculate current entropy based on the dispersion of "1"s
    grid = np.array(grid)
    total_cells = grid.size
    occupied_cells = np.sum(grid == 1)
    probability_occupied = occupied_cells / total_cells
    probability_empty = 1 - probability_occupied
    
    # Standard Shannon entropy calculation
    if probability_occupied == 0 or probability_empty == 0:
        entropy = 0
    else:
        entropy = - (probability_occupied * np.log2(probability_occupied) + 
                     probability_empty * np.log2(probability_empty))
    
    # Normalize by the log of the total grid size (dynamic phase space normalization)
    normalized_entropy = entropy / np.log2(total_cells + 1e-10)
    return normalized_entropy

def dispersion(grid, alpha=0.5, beta=0.5):
    # Calculate normalized entropy
    density_entropy = normalized_entropy(grid)
    
    # Calculate dispersion relative to center
    dispersion_entropy = mean_distance(grid, 50,50)
    
    # Combined entropy measure
    combined_entropy = alpha * density_entropy + beta * dispersion_entropy
    return combined_entropy

def extract_active_region(grid):
    grid = np.array(grid)
    rows, cols = grid.shape
    # Find the indices where "1"s are located
    ones_indices = np.argwhere(grid == 1)
       
    # Find the bounds of the active region
    min_row, min_col = ones_indices.min(axis=0)
    max_row, max_col = ones_indices.max(axis=0)
    
    # Extract and return the bounding box containing all "1"s
    active_region = grid[min_row:max_row + 1, min_col:max_col + 1]
    return active_region

def variance_of_positions(grid):
    ones_positions = np.argwhere(grid == 1)
    
    # Calculate the variance in both x and y directions
    if len(ones_positions) > 0:
        x_var = np.var(ones_positions[:, 0])
        y_var = np.var(ones_positions[:, 1])
        return x_var + y_var
    return 0

#def dispersion(state):
#    mean_distance_state = mean_distance(state)
#    variance_state = variance_of_positions(state)
#    
#    return mean_distance_state * np.sqrt(variance_state + 1e-10)  # Add a small constant to avoid log(0)
    

def expand_grid(grid, expansion_size = 1):
    grid = np.array(grid)
    original_shape = grid.shape
    
    new_shape = (original_shape[0] + 2 * expansion_size, original_shape[1] + 2 * expansion_size)
    expanded_grid = np.zeros(new_shape, dtype=grid.dtype)

    # Place the original grid in the center of the new expanded grid
    expanded_grid[expansion_size:expansion_size + original_shape[0], 
    expansion_size:expansion_size + original_shape[1]] = grid

    return expanded_grid
                
            

def local_density_macrostate(vector, block_size=5):
    """
    Divide the 1D array (vector) into smaller blocks and determine densities.
    Create a macrostate based on the distribution of densities.
    """
    vector_length = len(vector)
    local_densities = []
    
    # Divide the 1D vector into smaller blocks and calculate the density
    for i in range(0, vector_length, block_size):
        block = vector[i:i + block_size]
        density = np.sum(block) / len(block)  # Density is the sum of "1"s divided by block size
        local_densities.append(density)
    
    # Aggregate local densities into a histogram to define a macrostate
    density_histogram, _ = np.histogram(local_densities, bins=np.linspace(0, 1, 5))  # Example: 5 density ranges
    return tuple(density_histogram)  # Macrostate is defined by the density histogram

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
        

    
def submatrix1d(grid): #Define submatrix function on the grid to create the dependency vectors
    rows = len(grid)-2 #Runs over all values in vector minus the padding
    submatrices = [] #Initializes vector
    for i in range(rows):
        submatrices.append(grid[i:i+3]) #appends with a vector of the cell and its neighbors
        #print(submatrices)
    return submatrices #Returns a list of these vectors

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
    if dimension == 1: #1d case
        Kernel = np.ones(kernel_width, dtype = int)#Initializes a kernel of ones to kind of convolve (but not really) with the submatrices
        Kernel.tolist()
        #print(Kernel)
        value_state_pad = np.pad(value_state, (1), 'constant', constant_values=0) #pads the value state vector
        System_StateN = value_state #Sets the system state to the value state
        #print(f"systemstaten: {value_state}")
        System_StateNP = submatrix1d(value_state_pad) #assigns System_StateNP to the list of submatrices
        
        if dependency == "P": #If the automata is position dependent
            for i in range(len(System_StateN)): #For each index of the value state
                result = np.multiply(Kernel, System_StateNP[i]) #Take the element wise product of it with the vector of 1s
                System_StateN[i] = result.astype(int).tolist() #Make a list of all these products, these are each neighborhood
        elif dependency == "S": #If automata is state dependent
            # State-dependent automata
            for k in range(len(System_StateN)): #For each index in the system state
                if k == 0: #set the first value to be the value of the cell itself
                    System_StateN[k] = System_StateNP[0]  
                else:
                    result = np.multiply(Kernel, System_StateNP[k - 1]) #Otherwise set it to be the value of the neighbors
                    System_StateN[k] = result.astype(int).tolist()
          
    elif dimension == 2:  # 2D case
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
    
def partitions_by_birth1d(rule):
    partitions = {0:[], 1:[]}
    binaries = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    
    for i, vector in enumerate(binaries): #For each vector in the system state (dependency form)
        for symbol in vector: #for each symbol, 0 or 1 in the vector
            rule.transition(symbol) #Permorm the transition function on that 1 or 0
                # print(f"Current state: {Rule_32.current_state}")
        if rule.current_state[0] == "A": #If the state is an A type, the value vector at that index is 1
            partitions[1].append(vector)
        elif rule.current_state[0] == "D": #If a D type, then the vector at this index is 0
            partitions[0].append(vector)
        rule.reset() #Reset the automata for next vector
    return partitions        

def partitions_by_sum1d():
    partitions = {0:[[0,0,0]], 1:[[1,0,0],[0,1,0],[0,0,1]], 2:[[1,1,0],[1,0,1],[0,1,1]], 3:[[1,1,1]]}
    return partitions

def partitions_by_all1d():
    partitions = {0:[[0,0,0]], 1:[[0,0,1]], 2:[[0,1,0]], 3:[[0,1,1]], 4:[[1,0,0]], 5:[[1,0,1]], 6:[[1,1,0]], 7:[[1,1,1]]}
    return partitions

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
    
def find_highlife_replicator(grid):
    replicator_pattern = np.array([
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0]
    ])
    
    # Convert grid to a numpy array if it’s not already
    grid = np.array(grid)
    
    # Check if grid matches the exact size of replicator_pattern
    if grid.shape == replicator_pattern.shape:
        if np.array_equal(grid, replicator_pattern):
            return 1  # One replicator found if they match
    
    # Otherwise, continue with previous logic if grid size is larger than 5x5
    replicator_count = 0
    rows, cols = grid.shape
    sub_rows, sub_cols = replicator_pattern.shape
    
    for i in range(rows - sub_rows + 1):
        for j in range(cols - sub_cols + 1):
            # Extract the submatrix and convert it to a numpy array
            submatrix = grid[i:i + sub_rows, j:j + sub_cols]
            #print(f"{submatrix}=={replicator_pattern}")
            if np.array_equal(submatrix, replicator_pattern):
                print("YAY")
                replicator_count += 1
                print(replicator_count)
    return replicator_count

def lempel_ziv_complexity(binary_sequence):
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

def Sequence_States(State, num_steps, rule, dimension): 
    machine_epsilon = np.finfo(float).eps
    data_set = []  # Initialize an empty list to store the data
    entropy_list = []  # List to store entropy values over time
    partitions_birth1d = partitions_by_birth1d(rule)
    partitions_sum1d = partitions_by_sum1d()
    partitions_all1d = partitions_by_all1d()
  #  partitions_all2d = partitions_by_all2d()
    #print(f"partitions: {partitions_all2d}")
    phase = set((pos, tuple(config)) for pos, config in enumerate(State))
    initial_entropy = math.log2(len(phase))
    k = 1 / initial_entropy
    print(k)
    entropy_grow = 0
    for t in range(num_steps): 
        start_time = time.time()
        data = []  # Initialize dataset for this time step
        value_stateN = next_state(State, 5, 2, rule) # Get next state from FSM
        value_stateN1 = extract_active_region(value_stateN)
        #print(f"value state for complexity: {value_stateN}")
        end_time = time.time()
        if dimension == 1:
            complexity_lz2d = 0
            complexity_k2d = 0
            complexity_lz1d = lempel_ziv_complexity(value_stateN)
            complexity_k1d = kolmogorov_complexity(value_stateN)
            #print(f"complexity: {complexity_lz1d}")
        elif dimension == 2:
            complexity_lz1d = 0
            complexity_k1d = 0
            value_stateConcat = list(itertools.chain.from_iterable(value_stateN))
            complexity_lz2d = lempel_ziv_complexity(value_stateConcat)   
            complexity_k2d = kolmogorov_complexity(value_stateConcat)                   
        print(f"Next state after FSM at t = {t+1}: {value_stateN}")
        replicator_count = find_highlife_replicator(value_stateN)
        print(replicator_count)
        DependencyN = dependency_form(value_stateN, 3, "S", 2)  # Get dependency form of the value state
        Phase = dependency_form(value_stateN1, 3, "S", 2)
        new_phase_entries = set((pos, tuple(config)) for pos, config in enumerate(Phase))
        phase.update(new_phase_entries)
        boltzmann_entropy = k * math.log2(len(phase))
        print(boltzmann_entropy)
        #print(f"DependecnyN: {DependencyN}")
        # Update the state for the next iteration
        State = DependencyN
        #print(f"dependencyn: {DependencyN}")
        #print(f"State: {State}")

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
        
        #print(f"Entropy of the phase space at t = {t}: {entropy}")
        # Store the entropy value for this time step    
        partition_count_birth = {0:0, 1:0}
        partition_count_sum = {0:0, 1:0, 2:0, 3:0}
        partition_count_all = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
        partition_count_all2d = {}
        #partition_count_all2d = {i:0 for i in range(512)}
        #print(partition_count_all2d)
        
        for vector in State:
            if vector in partitions_birth1d[0]:
                partition_count_birth[0] += 1
            else:
                partition_count_birth[1] += 1
                
            if vector in partitions_sum1d[0]:
                partition_count_sum[0] += 1
            elif vector in partitions_sum1d[1]:
                partition_count_sum[1] += 1
            elif vector in partitions_sum1d[2]:
                partition_count_sum[2] += 1            
            elif vector in partitions_sum1d[3]:
                partition_count_sum[3] += 1
                
            if vector in partitions_all1d[0]:
                partition_count_all[0] += 1
            if vector in partitions_all1d[1]:
                partition_count_all[1] += 1            
            if vector in partitions_all1d[2]:
                partition_count_all[2] += 1                
            if vector in partitions_all1d[3]:
                partition_count_all[3] += 1
            if vector in partitions_all1d[4]:
                partition_count_all[4] += 1
            if vector in partitions_all1d[5]:
                partition_count_all[5] += 1
            if vector in partitions_all1d[6]:
                partition_count_all[6] += 1
            if vector in partitions_all1d[7]:
                partition_count_all[7] += 1
                
            # Increment the count for the partition corresponding to the integer_value
            #for i in range(512):
                #print(f"if {vector} == {partitions_all2d[i]}")
             #   if vector == partitions_all2d[i]:
              #      partition_count_all2d[i] += 1
                    #print("YAY")
                    #print(f"partitioncount{i} = {partition_count_all2d[i]}")
               #     break
        #print(partition_count_all2d)
            macrostate = local_density_macrostate(vector)
            if macrostate in partition_count_all2d:
                partition_count_all2d[macrostate] += 1
            else:
                partition_count_all2d[macrostate] = 1
        
                      
        partition_summation_birth = partition_count_birth[0] + partition_count_birth[1]
        partition_birth_prob0 = partition_count_birth[0] / (partition_summation_birth + machine_epsilon)
        partition_birth_prob1 = partition_count_birth[1] / (partition_summation_birth + machine_epsilon)
        
        entropy_partitions_birth = -(partition_birth_prob0 * np.log2(partition_birth_prob0 + 1e-10) + partition_birth_prob1 * np.log2(partition_birth_prob1 + 1e-10))
        KS_entropy_birth = (1/(np.log(t + machine_epsilon)))*entropy_partitions_birth
        
        partition_summation_sum = partition_count_sum[0] + partition_count_sum[1] + partition_count_sum[2]+partition_count_sum[3]
        partition_sum_prob0 = partition_count_sum[0] / (partition_summation_sum + machine_epsilon)
        partition_sum_prob1 = partition_count_sum[1] / (partition_summation_sum + machine_epsilon)
        partition_sum_prob2 = partition_count_sum[2] / (partition_summation_sum + machine_epsilon)
        partition_sum_prob3 = partition_count_sum[3] / (partition_summation_sum + machine_epsilon)       
        
        partition_sum_probs = np.array([partition_sum_prob0, partition_sum_prob1, partition_sum_prob2, partition_sum_prob3])
        entropy_partitions_sum = -np.sum(partition_sum_probs * np.log2(partition_sum_probs + 1e-10))
        KS_entropy_sums = (1/(np.log(t + machine_epsilon)))*entropy_partitions_sum
        
        partition_summation_all = sum(partition_count_all.values())
        partition_all_prob0 = partition_count_all[0] / (partition_summation_all + machine_epsilon)
        partition_all_prob1 = partition_count_all[1] / (partition_summation_all + machine_epsilon)        
        partition_all_prob2 = partition_count_all[2] / (partition_summation_all + machine_epsilon)        
        partition_all_prob3 = partition_count_all[3] / (partition_summation_all + machine_epsilon)        
        partition_all_prob4 = partition_count_all[4] / (partition_summation_all + machine_epsilon)
        partition_all_prob5 = partition_count_all[5] / (partition_summation_all + machine_epsilon)
        partition_all_prob6 = partition_count_all[6] / (partition_summation_all + machine_epsilon)
        partition_all_prob7 = partition_count_all[7] / (partition_summation_all + machine_epsilon)
        
        partition_all_probs = np.array([partition_all_prob0, partition_all_prob1, partition_all_prob2, partition_all_prob3, partition_all_prob4, partition_all_prob5, partition_all_prob6, partition_all_prob7])
        entropy_partitions_all = -np.sum(partition_all_probs * np.log2(partition_all_probs + 1e-10))
        KS_entropy_all = (1/(np.log(t + machine_epsilon)))*entropy_partitions_all
        print(partition_count_all )
        print(partition_all_probs)
        
        # Calculate the sum of all partition counts
        partition_summation_all2d = sum(partition_count_all2d.values())
        
        # Initialize an array to store probabilities for each partition
        partition_all2d_prob = np.zeros(512)
        
        # Calculate probability for each partition
        
        # Compute entropy
        entropy_partitions_all2d = -np.sum(partition_all2d_prob * np.log2(partition_all2d_prob + 1e-10))
        
        # Calculate KS entropy with a log scaling
        KS_entropy_all2d = (1 / (np.log(t + machine_epsilon))) * entropy_partitions_all2d
        # Calculate Boltzmann Entropy for each macrostate
        partition_summation_all2d = sum(partition_count_all2d.values())

        if partition_summation_all2d > 0:
            # Initialize a list to store probabilities for each partition that actually exists
            partition_all2d_prob = []
        
            # Calculate probabilities only for existing keys in the dictionary
            for key, count in partition_count_all2d.items():
                probability = count / (partition_summation_all2d + machine_epsilon)
                partition_all2d_prob.append(probability)
        
            # Convert list to numpy array for entropy calculation
            partition_all2d_prob = np.array(partition_all2d_prob)
        
            # Compute entropy using only the probabilities that are present
            entropy_partitions_all2d = -np.sum(partition_all2d_prob * np.log2(partition_all2d_prob + 1e-10))
        
            # Calculate KS entropy with a log scaling
            KS_entropy_all2d = (1 / (np.log(t + machine_epsilon))) * entropy_partitions_all2d

            
            # Initialize Boltzmann entropy
            #boltzmann_entropy = 0
            #print(f"partition_count: {partition_count_all2d}")
        
            # Compute entropy across all macrostates
            #for macro, count in partition_count_all2d.items():
             #   if count > 0:
              #      omega = count
               #     boltzmann_entropy += np.log(omega)
        
            # Normalize by total to represent phase space disorder
           # boltzmann_entropy /= partition_summation_all2d
            
        dispersion_state = dispersion(value_stateN)

        step_duration = end_time - start_time
        
        
        entropy_list.append({
            'time': t,
            'entropy_states': entropy,
            'entropy_partitions_birth': entropy_partitions_birth,
            'entropy_partitions_sum': entropy_partitions_sum,
            'entropy_partitions_all': entropy_partitions_all,
            'entropy_partitions_all2d': entropy_partitions_all2d,
            'complexity_serieslz1d': complexity_lz1d,
            'complexity_seriesk1d' : complexity_k1d,
            'complexity_serieslz2d': complexity_lz2d,
            'complexity_seriesk2d' : complexity_k2d,
            'KS_entropy_birth':KS_entropy_birth,
            'KS_entropy_sums':KS_entropy_sums,
            'KS_entropy_all':KS_entropy_all,
            'KS_entropy_all2d' : KS_entropy_all2d,
            'Highlife_replicators' : replicator_count,
            "time_per_step": step_duration,
            "entropy_grow" : entropy_grow,
            'boltzmann_entropy': boltzmann_entropy,
            'dispersion': dispersion_state
            
            })

    
    # Return the data set as a DataFrame and the entropy evolution over time
    return entropy_list


# Example usage
Value_State_norm = [0]*2001
Value_State_norm[1001] = 1
#print(f"Initial state at t = 0:        {Value_State_norm}")

value_state_stochastic = []
for i in range(501):
    random_digit = random.randint(0, 1)
    value_state_stochastic.append(random_digit)

# Create a 2D array for the glider
rows, cols = 10, 10  # Example dimensions
value_state2dglid = [[0 for _ in range(cols)] for _ in range(rows)]  # Properly initialize 2D array
middle_row = rows // 2
middle_col = cols // 2

# Define the glider shape
value_state2dglid[middle_row-1][middle_col] = 1
value_state2dglid[middle_row][middle_col+1] = 1
value_state2dglid[middle_row+1][middle_col-1] = 1
value_state2dglid[middle_row+1][middle_col] = 1
value_state2dglid[middle_row+1][middle_col+1] = 1
#print(f"Next state after FSM at t = 0 {value_state2dglid}")

rows, cols = 50, 50  # Example dimensions
value_state2drep = [[0 for _ in range(cols)] for _ in range(rows)]  # Properly initialize 2D array
middle_row = rows // 2
middle_col = cols // 2

# Define the replicator shape
value_state2drep[middle_row-2][middle_col] = 1
value_state2drep[middle_row-2][middle_col+1] = 1
value_state2drep[middle_row-2][middle_col+2] = 1
value_state2drep[middle_row-1][middle_col-1] = 1
value_state2drep[middle_row-1][middle_col+2] = 1
value_state2drep[middle_row][middle_col-2] = 1
value_state2drep[middle_row][middle_col+2] = 1
value_state2drep[middle_row+1][middle_col-2] = 1
value_state2drep[middle_row+1][middle_col+1] = 1
value_state2drep[middle_row+2][middle_col-2] = 1
value_state2drep[middle_row+2][middle_col-1] = 1
value_state2drep[middle_row+2][middle_col] = 1
print(f"Next state after FSM at t = 0 {value_state2drep}")


def random_inside_larger(outer_size, inner_size):
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
    
value_state_stoch2d = random_inside_larger(150, 50)        

        



        

# Example function calls (ensure the dependency_form function is defined correctly)
#Test_Dependency1 = dependency_form(Value_State_norm, 3, "P", 1) 
#Test_Dependency2 = dependency_form(value_state_stochastic, 3, "P", 1)
#Test_Dependency2D = dependency_form(value_state2dglid, 3, "S", 2) #GLIDER
#Test_Dependency2D = dependency_form(value_state_stoch2d, 3, "S", 2) 
Test_Dependency2D = dependency_form(value_state_stoch2d, 3, "S", 2) 

#print(f"Test_dependency2d: {Test_Dependency2D}")

# Generate data and calculate entropy over time
entropy_list = Sequence_States(Test_Dependency2D, 350, High_life,2)

#Print the final entropy evolution
#print(entropy_list)

time = [entry['time'] for entry in entropy_list]
entropy_values_states = [entry['entropy_states'] for entry in entropy_list]
entropy_birth_partitions = [entry['entropy_partitions_birth'] for entry in entropy_list]
entropy_sum_partitions = [entry['entropy_partitions_sum'] for entry in entropy_list]
entropy_all_partitions = [entry['entropy_partitions_all'] for entry in entropy_list]
complexity_serieslz1d =  [entry['complexity_serieslz1d'] for entry in entropy_list]
entropy_all_partitions2d = [entry['entropy_partitions_all2d'] for entry in entropy_list]
complexity_seriesk1d =  [entry['complexity_seriesk1d'] for entry in entropy_list]
complexity_serieslz2d =  [entry['complexity_serieslz2d'] for entry in entropy_list]
complexity_seriesk2d =  [entry['complexity_seriesk2d'] for entry in entropy_list]
KS_entropy_births = [entry['KS_entropy_birth'] for entry in entropy_list]
KS_entropy_sums = [entry['KS_entropy_sums'] for entry in entropy_list]
KS_entropy_all = [entry['KS_entropy_all'] for entry in entropy_list]
KS_entropy_all2d = [entry['KS_entropy_all2d'] for entry in entropy_list]
highlife_replicators = [entry['Highlife_replicators'] for entry in entropy_list]
time_per_step = [entry['time_per_step'] for entry in entropy_list] 
entropy_grow = [entry['entropy_grow'] for entry in entropy_list] 
boltzmann_entropy = [entry['boltzmann_entropy'] for entry in entropy_list] 
dispersions_state = [entry['dispersion'] for entry in entropy_list] 





df_entropy = pd.DataFrame({
    'time': time,
    'Entropy_States': entropy_values_states,
    'Entropy_birth_partitions': entropy_birth_partitions,
    'Entropy_sum_partitions': entropy_sum_partitions,
    'Entropy_all_partitions': entropy_all_partitions,
    'Entropy_all_partitions2d': entropy_all_partitions2d,
    'complexity_serieslz1d': complexity_serieslz1d,
    'complexity_seriesk1d': complexity_seriesk1d,
    'complexity_serieslz2d': complexity_serieslz2d,
    'complexity_seriesk2d': complexity_seriesk2d,
    'KS_entropy_births':KS_entropy_births,
    'KS_entropy_sums':KS_entropy_sums,
    'KS_entropy_all':KS_entropy_all,
    'KS_entropy_all2d':KS_entropy_all2d,
    'Highlife_replicators':highlife_replicators,
    'Time_per_step':time_per_step,
    'entropy_grow':entropy_grow,
    'boltzmann_entropy':boltzmann_entropy,
    'dispersion':dispersions_state
    
})

def shannon_entropy(values):
    # Get the unique values and their counts
    unique_values, counts = np.unique(values, return_counts=True)
    
    # Calculate the probability distribution
    probabilities = counts / len(values)
    
    # Calculate Shannon entropy with a small offset to avoid log(0)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# Function to calculate entropy of entropy (Shannon entropy growth over time)
def entropy_of_entropy(df, column, new_column):
    entropy_two = []
    
    # Iterate through the column and calculate entropy as more elements are added
    for i in range(1, len(df[column]) + 1):
        entropy_n = shannon_entropy(df[column][:i])  # Pass in first i elements
        entropy_two.append(entropy_n)  # Append the entropy value for this slice
    
    # Add the results as a new column to the existing DataFrame
    df[new_column] = entropy_two
    return df

# Example usage to calculate entropy of entropy for 'Entropy_States' column
df_entropy = entropy_of_entropy(df_entropy, 'Entropy_States', 'Entropy_of_States')
df_entropy = entropy_of_entropy(df_entropy, 'Entropy_birth_partitions', 'Entropy_of_Birth_Partitions')
df_entropy = entropy_of_entropy(df_entropy, 'Entropy_sum_partitions', 'Entropy_of_Sum_Partitions')
df_entropy = entropy_of_entropy(df_entropy, 'Entropy_all_partitions', 'Entropy_of_All_Partitions')


#Provide the full file path to save on Desktop
file_path = r'C:\Users\benja\OneDrive\Desktop\HighLifeEntropy.xlsx' 

# Save the DataFrame to an Excel file
df_entropy.to_excel(file_path, index=False)

print(f'File saved to: {file_path}')
