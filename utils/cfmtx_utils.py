import numpy as np

# Transforms a patterns dict to a flattened list
def preprocess_patterns(patterns):

    pattern_list = []
    for i in range(len(patterns)):
        pattern = [int(x) for x in patterns[i]]
        pattern_list.append(pattern)

    return pattern_list

def create_array_offsets(array):
    
    offsets = np.zeros(len(array) + 1, dtype=np.int32)
    for i in range(len(array)):
        offsets[i + 1] = offsets[i] + len(array[i])

    return offsets

def next_power_of_two(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()

def flatten_array(array):
    flattened = []
    for i in range(len(array)):
        for j in range(len(array[i])):
            flattened.append(array[i][j])

    return flattened