time = [entry['time'] for entry in entropy_list]
entropy_values_states = [entry['entropy_states'] for entry in entropy_list]
entropy_birth_partitions = [entry['entropy_partitions_birth'] for entry in entropy_list]
entropy_sum_partitions = [entry['entropy_partitions_sum'] for entry in entropy_list]
entropy_all_partitions = [entry['entropy_partitions_all'] for entry in entropy_list]
complexity_serieslz1d =  [entry['complexity_serieslz1d'] for entry in entropy_list]
complexity_seriesk1d =  [entry['complexity_seriesk1d'] for entry in entropy_list]
complexity_serieslz2d =  [entry['complexity_serieslz2d'] for entry in entropy_list]
complexity_seriesk2d =  [entry['complexity_seriesk2d'] for entry in entropy_list]
KS_entropy_births = [entry['KS_entropy_birth'] for entry in entropy_list]
KS_entropy_sums = [entry['KS_entropy_sums'] for entry in entropy_list]
KS_entropy_all = [entry['KS_entropy_all'] for entry in entropy_list]

df_entropy = pd.DataFrame({
    'time': time,
    'Entropy_States': entropy_values_states,
    'Entropy_birth_partitions': entropy_birth_partitions,
    'Entropy_sum_partitions': entropy_sum_partitions,
    'Entropy_all_partitions': entropy_all_partitions,
    'complexity_serieslz1d': complexity_serieslz1d,
    'complexity_seriesk1d': complexity_seriesk1d,
    'complexity_serieslz2d': complexity_serieslz2d,
    'complexity_seriesk2d': complexity_seriesk2d,
    'KS_entropy_births':KS_entropy_births,
    'KS_entropy_sums':KS_entropy_sums,
    'KS_entropy_all':KS_entropy_all
    
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
