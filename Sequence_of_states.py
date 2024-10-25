def Sequence_States(State, num_steps, rule, dimension): 
    machine_epsilon = np.finfo(float).eps
    data_set = []  # Initialize an empty list to store the data
    entropy_list = []  # List to store entropy values over time
    partitions_birth = partitions_by_birth1d(rule)
    partitions_sum = partitions_by_sum1d()
    partitions_all = partitions_by_all1d()
    
    for t in range(num_steps):  
        data = []  # Initialize dataset for this time step
        value_stateN = next_state(State, 5, 2, rule) # Get next state from FSM
        print(f"value state for complexity: {value_stateN}")
        if dimension == 1:
            complexity_lz2d = 0
            complexity_k2d = 0
            complexity_lz1d = lempel_ziv_complexity(value_stateN)
            complexity_k1d = kolmogorov_complexity(value_stateN)
            print(f"complexity: {complexity_lz1d}")
        elif dimension == 2:
            complexity_lz1d = 0
            complexity_k1d = 0
            value_stateConcat = list(itertools.chain.from_iterable(value_stateN))
            complexity_lz2d = lempel_ziv_complexity(value_stateConcat)   
            complexity_k2d = kolmogorov_complexity(value_stateConcat)                   
        print(f"Next state after FSM at t = {t+1}: {value_stateN}")
        DependencyN = dependency_form(value_stateN, 3, "S", 2)  # Get dependency form of the value state
        #print(f"DependecnyN: {DependencyN}")
        # Update the state for the next iteration
        State = DependencyN
        #print(State)

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
        #print(f"Entropy of the phase space at t = {t}: {entropy}")
        
        # Store the entropy value for this time step    
        partition_count_birth = {0:0, 1:0}
        partition_count_sum = {0:0, 1:0, 2:0, 3:0}
        partition_count_all = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
        
        for vector in State:
            if vector in partitions_birth[0]:
                partition_count_birth[0] += 1
            else:
                partition_count_birth[1] += 1
                
            if vector in partitions_sum[0]:
                partition_count_sum[0] += 1
            elif vector in partitions_sum[1]:
                partition_count_sum[1] += 1
            elif vector in partitions_sum[2]:
                partition_count_sum[2] += 1            
            elif vector in partitions_sum[3]:
                partition_count_sum[3] += 1
                
            if vector in partitions_all[0]:
                partition_count_all[0] += 1
            if vector in partitions_all[1]:
                partition_count_all[1] += 1            
            if vector in partitions_all[2]:
                partition_count_all[2] += 1                
            if vector in partitions_all[3]:
                partition_count_all[3] += 1
            if vector in partitions_all[4]:
                partition_count_all[4] += 1
            if vector in partitions_all[5]:
                partition_count_all[5] += 1
            if vector in partitions_all[6]:
                partition_count_all[6] += 1
            if vector in partitions_all[7]:
                partition_count_all[7] += 1

                      
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
        
        entropy_list.append({
            'time': t,
            'entropy_states': entropy,
            'entropy_partitions_birth': entropy_partitions_birth,
            'entropy_partitions_sum': entropy_partitions_sum,
            'entropy_partitions_all': entropy_partitions_all,
            'complexity_serieslz1d': complexity_lz1d,
            'complexity_seriesk1d' : complexity_k1d,
            'complexity_serieslz2d': complexity_lz2d,
            'complexity_seriesk2d' : complexity_k2d,
            'KS_entropy_birth':KS_entropy_birth,
            'KS_entropy_sums':KS_entropy_sums,
            'KS_entropy_all':KS_entropy_all
            
            })
    
    # Return the data set as a DataFrame and the entropy evolution over time
    return entropy_list
