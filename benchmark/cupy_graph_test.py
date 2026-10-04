import cupy as cp
from cupyx.scipy.sparse import csr_matrix


def simple_sptrsv_csr(L, b):
    n = L.shape[0]
    x = cp.zeros(n)
    for i in range(n):
        x[i] = b[i]
        # Begin stream capture
        start = L.indptr[i].item()
        end = L.indptr[i + 1].item()-1
        for j in range(start, end):
            x[i] -= L.data[j] * x[L.indices[j]]
        x[i] /= L.data[L.indptr[i + 1] - 1]
    return x


#Create a non-blocking stream for capturing operations,
s1 = cp.cuda.Stream(non_blocking=True)


# Define a 4x4 lower triangular sparse matrix
data = cp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
indices = cp.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
indptr = cp.array([0, 1, 3, 6, 10], dtype=cp.int64)
L = csr_matrix((data, indices, indptr), shape=(4, 4))

# Define the right-hand side vector
b = cp.array([1.0, 2.0, 3.0, 4.0])

correct_sol = simple_sptrsv_csr(L, b)
print(correct_sol)

# Result placeholder
x = cp.empty_like(b)

graph_list = []

# with s1:
#     for i in range(3):
#         # Begin stream capture
#         start = L.indptr[i].item()
#         end = L.indptr[i + 1].item()-1
#
#         s1.begin_capture()
#         x[i] = b[i]  # Copy b to x
#         for j in range(start, end):
#             x[i] -= L.data[j] * x[L.indices[j]]
#         x[i] /= L.data[end - 1]
#         # End stream capture and obtain the CUDA graph
#         graph = s1.end_capture()
#         graph_list.append(graph)


with s1:
    # Iteration 0
    s1.begin_capture()
    x[0] = b[0]  # Copy b[0] to x[0]
    x[0] /= L.data[L.indptr[1] - 1]
    graph = s1.end_capture()
    graph_list.append(graph)

    # Iteration 1
    s1.begin_capture()
    x[1] = b[1]  # Copy b[1] to x[1]
    x[1] -= L.data[1] * x[L.indices[1]]
    x[1] /= L.data[L.indptr[2] - 1]
    graph = s1.end_capture()
    graph_list.append(graph)

    # Iteration 2
    s1.begin_capture()
    x[2] = b[2]  # Copy b[2] to x[2]
    x[2] -= L.data[3] * x[L.indices[3]]
    x[2] -= L.data[4] * x[L.indices[4]]
    x[2] /= L.data[L.indptr[3] - 1]
    graph = s1.end_capture()
    graph_list.append(graph)

    # Iteration 3
    s1.begin_capture()
    x[3] = b[3]  # Copy b[3] to x[3]
    x[3] -= L.data[6] * x[L.indices[6]]
    x[3] -= L.data[7] * x[L.indices[7]]
    x[3] -= L.data[8] * x[L.indices[8]]
    x[3] /= L.data[L.indptr[4] - 1]
    graph = s1.end_capture()
    graph_list.append(graph)


#Launch the captured graph on the same stream,
print("Launching graph on s1:")
# Launch each graph in the list on the same stream
with s1:
    for graph in graph_list:
        graph.launch(stream=s1)
        # Synchronize the stream to ensure all graphs have completed execution
s1.synchronize()
print("Result after first launch:", x)
