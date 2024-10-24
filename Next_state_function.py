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
