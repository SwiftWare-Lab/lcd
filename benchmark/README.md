
# benchmark

## Table of operations

-------------------
| Operation  | Format         | Description |
|------------|----------------|-------------|
| sptrsv-fwd | CSR            | Forward substitution |
| sptrsv-bwd | CSR            | Backward substitution |
| ilu0       | CSR            | Incomplete LU factorization |
| ilu0-solve | CSR            | Solve using ILU0 factorization |
| ilu0       | CSC            | Incomplete LU factorization |
| Gauss-seidel | CSR            | Gauss-Seidel iteration |
| Cholesky   | CSC            | Cholesky factorization |
| Cholesky   | Supernodal CSC | Cholesky factorization |
| sptrsv-fwd | Supernodal CSC | Solve using Cholesky factorization |






Each operation is added as a separate benchmark. Each operation has a 
separate benchmark and implemetation file. The benchmark file is named 
`<operation>_benchmark.py` and the implementation file is named 
`<operation>.py`. 


The benchmark file loads a list of sparse matrices from the file using:
```python
mtx_list = get_matrix_list("/home/kazem/UFDB/spd_list.txt", "/home/kazem/UFDB/")
```
The list of matrices is stored in the file `spd_list.txt` and the matrices are stored in the directory `UFDB`. 

Or a few matrix paths can be hard coded as below
```python
mtx_list = [
    "/home/kazem/UFDB/Trefethen_20000b/Trefethen_20000b.mtx",
    "/home/kazem/Downloads/msc23052/msc23052.mtx",
    "/home/kazem/Downloads/olafu/olafu.mtx",
    "/home/kazem/Downloads/gyro_k/gyro_k.mtx",
    "/home/kazem/Downloads/cant/cant.mtx"
            ]
```

