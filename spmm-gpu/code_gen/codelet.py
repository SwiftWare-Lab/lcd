
indent = '    '


class AccessFunction:
    def __init__(self, name, relations, offset):
        self.name = name
        self.induction_variable = list(relations.keys())
        self.relations = relations
        self.offset = offset

    def get_pattern(self):
        affines = list([self.relations[var][-1] == 'affine' for var in self.induction_variable])
        preambles, access_pattern = f'', f''
        for i in range(len(affines)):
            if affines[i]:
                access_pattern += f'{self.relations[self.induction_variable[i]][0]} * {self.induction_variable[i]} +'
            else:
                tmp_name = f'{self.name}_{self.induction_variable[i]}'
                preambles += f'{indent}{tmp_name} = {self.relations[self.induction_variable[i]][:-1]} \n'
                access_pattern += f'{tmp_name}[{self.induction_variable[i]}] +'
        access_pattern += f' {self.offset}'
        return preambles, access_pattern


class Codelet:
    def __init__(self, name, iteration_space, access_funcs, data_spaces, operation="+="):
        self.name = name
        self.iteration_space = iteration_space
        self.access_funs = access_funcs
        self.data_spaces = data_spaces
        self.operation = operation

    def gen_code(self):
        iteration_vars = list(self.iteration_space.keys())
        # affine if self[iterations_ver][-1] == 'affine'
        affines = all([self.iteration_space[var][-1] == 'affine' for var in iteration_vars])

        loop_nest = f''
        # generate loops
        for k, v in self.iteration_space.items():
            loop_nest += f'{indent}for {k} in range({v[0]}, {v[1]}, {v[2]}):\n'

        # generate data spaces and access functions
        operations, preamble = f'', f''
        rhs, lhs = f'', f''
        for i, access_func in enumerate(self.access_funs):
            preambles, access_pattern = access_func.get_pattern()
            preamble += preambles
            if i == 0: # left hand side
                lhs += f'{indent}{self.data_spaces[access_func.name]}[{access_pattern}] \n'
            else: # right hand side
                rhs += f'{indent}{self.data_spaces[access_func.name]}[{access_pattern}] {self.operation}\n'
        operations = lhs + '=' + rhs
        op_indent = f''
        for i in range(len(self.access_funs)):
            op_indent += f'{indent}'
        operations = op_indent + operations
        return preamble + loop_nest + operations


# this is how I generated code for this codelet:
# Cols = [1, 5, 6]
# Rows = [0, 2, 3]
# for i0 in range(0, 3, 1):
#     for i1 in range(0, 3, 1):
#         for i2 in range(0, 3, 1):
#             C[i0 * 1 + Rows[i2] ] += A[Rows[i2] + i1] * B[Cols[i2] + i1]

# create an array of dict
f = {'i0': [1, 'affine'], 'i1': [0, 'affine'], 'i2': [0, 2, 3, 'non-affine']}
g = {'i0': [0, 'affine'], 'i1': [1, 'affine'], 'i2': [3, 'affine']}
h = {'i0': [0, 'affine'], 'i1': [1, 'affine'], 'i2': [1, 5, 8, 'non-affine']}
af = AccessFunction('f', f, 0)
ag = AccessFunction('g', g, 0)
ah = AccessFunction('h', h, 0)
pre, acc = af.get_pattern()
iter_space = {'i0': [0, 3, 1], 'i1': [0, 3, 1], 'i2': [0, 3, 1]}
data_spaces = {'f': 'C', 'g': 'A', 'h': 'B'}

c = Codelet('codelet', iter_space, [af, ag, ah], data_spaces)
print(c.gen_code())
