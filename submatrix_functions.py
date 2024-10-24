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
