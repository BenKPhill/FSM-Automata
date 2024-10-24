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
