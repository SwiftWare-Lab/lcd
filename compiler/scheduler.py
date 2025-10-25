

# DAG is adjacency list of the dependency DAG.
# DAG_rev is the same DAG with reversed edges.
# l1_nodes_num: number of elements in the first level of the DAG.
# Outputs: level_ptr, partition_ptr, id

def executer(Ap, Ai, Ax, **kwargs):
    assert 'schedule' in kwargs
    schedule = kwargs['schedule']
    for s in schedule:
        for first, second in s:
            for i1 in first:
                pass
            for j1 in second:
                pass


def chunks_part(lst, n):
    for i in range(0, len(lst), n):
        yield [], lst[i:i + n]


def schedule_bipartite(DAG: list[list], num_threads):
    init_block_size = 32
    l1_nodes_num = len(DAG) #TODO: I considered same number of computations for first level and second level for now. Need a fix for future.
    num_partitions = max(num_threads * 2, l1_nodes_num // init_block_size)
    new_block_size = l1_nodes_num // num_partitions
    residue = l1_nodes_num % num_partitions
    schedule = [([],[]) for _ in range(num_partitions)]
    unfused_elements = []
    for p in range(num_partitions):
        if p >= residue:
            tile_range = range(p*new_block_size + residue, (p+1)*new_block_size + residue)
        else:
            tile_range = range(p*new_block_size + p, (p+1)*new_block_size + p + 1)
        schedule[p][0].extend(tile_range)
        for l2n in tile_range:
            if len(DAG[l2n]) > 0 and DAG[l2n][0] >= tile_range[0] and DAG[l2n][-1] <= tile_range[-1]:
                schedule[p][1].append(l2n)
            else:
                unfused_elements.append(l2n)
    num_new_partitions = num_threads * 2
    new_partitions = list(chunks_part(unfused_elements, num_new_partitions))
    schedule = [schedule, new_partitions]
    return schedule

