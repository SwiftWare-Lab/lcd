import cupy as cp
import numpy as np


scatter_kernel = cp.RawKernel(r'''
extern "C" __global__
void scatter_kernel(const double* update_vals, const int* row_indices, 
                   double* x, int n_rows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_rows) {
        int idx = row_indices[i];
        double val = update_vals[i];
        atomicAdd(&x[idx], -val);
    }
}
''', 'scatter_kernel')


class SupernodalSpTrsvCudaGraphSyncFree:
    def __init__(self, Ls, dag_matrix):
        self.Ls = Ls
        self.dag_matrix = dag_matrix
        
        self.d_Lx = cp.asarray(Ls["x"])
        self.d_Li = cp.asarray(Ls["i"], dtype=cp.int32)
        self.d_Lp = cp.asarray(Ls["p"], dtype=cp.int32)
        self.d_Lip = cp.asarray(Ls["ip"], dtype=cp.int32)
        self.d_sn_begin = cp.asarray(Ls["supernodes"], dtype=cp.int32)
        
        ncols = int(Ls["supernodes"][-1])
        self.d_x = cp.zeros(ncols, dtype=cp.float64)
        
        n_supernodes = len(Ls["supernodes"]) - 1
        self.n_supernodes = n_supernodes
        self.sn_metadata = []
        self.row_idx_per_sn = []
        
        for sn_id in range(n_supernodes):
            col_b = int(Ls["supernodes"][sn_id])
            col_e = int(Ls["supernodes"][sn_id + 1])
            sup_width = col_e - col_b
            sup_len = int(Ls["ip"][sn_id + 1] - Ls["ip"][sn_id])
            val_beg = int(Ls["p"][col_b])
            
            start = Ls["ip"][sn_id] + sup_width
            end = Ls["ip"][sn_id + 1]
            rows = cp.asarray(Ls["i"][start:end], dtype=cp.int32)
            self.row_idx_per_sn.append(rows)
            
            self.sn_metadata.append({
                'col_b': col_b,
                'col_e': col_e,
                'sup_width': sup_width,
                'sup_len': sup_len,
                'val_beg': val_beg,
                'n_scatter': len(rows)
            })
        
        self._extract_dependencies()
        self._build_graph()
    
    def _matvec_no_blas(self, A, x):
        # TODO (Da): Currently no cuBLAS, for graph capture
        return cp.sum(A * x[cp.newaxis, :], axis=1)
    
    def _trsv_lower(self, L, b):
        # TODO (Da): Currently no cuBLAS, for graph capture
        n = L.shape[0]
        for i in range(n):
            if i > 0:
                b[i] -= cp.sum(L[i, :i] * b[:i])
            b[i] /= L[i, i]
    
    def _extract_dependencies(self):
        self.dependencies = []
        dag_csr = self.dag_matrix.tocsr() if hasattr(self.dag_matrix, 'tocsr') else self.dag_matrix
        
        for sn_id in range(self.n_supernodes):
            deps = []
            for idx in range(dag_csr.indptr[sn_id], dag_csr.indptr[sn_id + 1]):
                dep_sn = dag_csr.indices[idx]
                if dep_sn != sn_id:
                    deps.append(int(dep_sn))
            self.dependencies.append(deps)
    
    def _process_supernode_operations(self, sn_id, stream):
        meta = self.sn_metadata[sn_id]
        col_b = meta['col_b']
        col_e = meta['col_e']
        sup_width = meta['sup_width']
        sup_len = meta['sup_len']
        val_beg = meta['val_beg']
        n_scatter = meta['n_scatter']
        
        with stream:
            Lx = self.d_Lx
            x = self.d_x
            
            diag_block = Lx[val_beg:val_beg + sup_len * sup_width].reshape(
                sup_len, sup_width, order="F"
            )
            x_block = x[col_b:col_e]
            
            L_diag = diag_block[:sup_width, :sup_width]
            cp.scipy.linalg.solve_triangular(L_diag, x_block, lower=True)
            # self._trsv_lower(L_diag, x_block)
            
            if sup_len > sup_width and n_scatter > 0:
                off_diag = diag_block[sup_width:, :]
                # update = self._matvec_no_blas(off_diag, x_block)
                update = cp.matmul(off_diag, x_block)
                
                row_idx = self.row_idx_per_sn[sn_id]
                threads_per_block = 256
                blocks = (n_scatter + threads_per_block - 1) // threads_per_block
                scatter_kernel(
                    (blocks,), (threads_per_block,),
                    (update, row_idx, x, cp.int32(n_scatter))
                )
        
        return stream
    
    def _topological_sort(self):
        in_degree = [len(self.dependencies[i]) for i in range(self.n_supernodes)]
        queue = [i for i in range(self.n_supernodes) if in_degree[i] == 0]
        sorted_order = []
        
        while queue:
            sn_id = queue.pop(0)
            sorted_order.append(sn_id)
            
            for other_sn in range(self.n_supernodes):
                if sn_id in self.dependencies[other_sn]:
                    in_degree[other_sn] -= 1
                    if in_degree[other_sn] == 0:
                        queue.append(other_sn)
        
        return sorted_order
    
    def _build_graph(self):
        # TODO (Da): single-stream capture for now; true concurrency needs lower-level APIs
        execution_order = self._topological_sort()
        main_stream = cp.cuda.Stream(non_blocking=True)
        
        main_stream.begin_capture()
        with main_stream:
            for sn_id in execution_order:
                self._process_supernode_operations(sn_id, main_stream)
        
        self._graph = main_stream.end_capture()
        self._launch_stream = main_stream
    
    def solve(self, b: np.ndarray):
        self.d_x.set(b)
        self._graph.launch(self._launch_stream)
        self._launch_stream.synchronize()
        return self.d_x.get()


def sptrsv_supernodal_cudagraph_syncfree(Ls, dag_matrix, b):
    solver = SupernodalSpTrsvCudaGraphSyncFree(Ls, dag_matrix)
    return solver.solve(b)

class SptrsvCudaGraphWavefront:
    def __init__(self, Ls, dag_matrix):
        self.Ls = Ls
        self.dag_matrix = dag_matrix