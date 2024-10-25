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
