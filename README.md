# gpu-backend

# compressed_format_generator.py

This file can be used to generate a compressed format version of the matrix.
To run this file the command below can be called

```
python compressed_format_generator.py <input_file> <kernel_m> <kernel_n>

example : python compressed_format_generator.py matrix.mtx 4 4
```

the input matrix should be .mtx format. Out put will be a file with .cfmtx with same name at same path.

# compressed_format_reader.py

This file can be used to read the compressed format version of the matrix
There is a function in this file called read_compressed_matrix which take
path of a .cfmtx file and return a compressed matrix object.

example :

```
path = "matrix.cfmtx"
compressed_matrix = read_compressed_matrix(path)
```