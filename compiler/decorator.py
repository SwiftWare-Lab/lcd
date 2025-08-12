import inspect


import types
import ast
from legality_check import finalize_legality_of_fusion, transform_ast, generate_dag_generation_code
from legality_check import extract_node_data, check_legality_of_fusion, extract_dag_creation_components
from spf_raising import SPFBuilder, TileFusionLoopBodyExtractor
import numpy as np



def fusion(**kwargs):
    def decorator(function: types.FunctionType):
        transformations = kwargs['transformation']
        function_props = kwargs['function_props']
        source = inspect.getsource(function)
        source = '\n'.join([l for l in source.splitlines() if not l.startswith("@")])

        old_code_obj = function.__code__
        cur_ast = ast.parse(source)
        for transformation in transformations:
            if finalize_legality_of_fusion(transformation, cur_ast, function_props):
                transformed_tree = transform_ast(transformation, cur_ast)
                print(ast.unparse(transformed_tree))
                transformed_tree = ast.fix_missing_locations(transformed_tree)
                cur_ast = transformed_tree
            else:
                raise "Transformation {} is not legal.".format(transformation)

        new_code_obj = compile(transformed_tree, old_code_obj.co_filename, 'exec')
        new_f = types.FunctionType(new_code_obj.co_consts[0], function.__globals__)
        return new_f
    return decorator


def sparse_fusion(**kwargs):
    def decorator(function: types.FunctionType):
        transformation = kwargs['transformation']
        num_threads = kwargs['num_threads'] # TODO: This should be either passed or get from the system
        source = inspect.getsource(function)
        source = '\n'.join([l for l in source.splitlines() if not l.startswith("@")])
        old_code_obj = function.__code__
        cur_ast = ast.parse(source)
        spf_builder = SPFBuilder()
        root_node = spf_builder.visit(cur_ast)
        node_data = extract_node_data(root_node)
        const, dependencies = check_legality_of_fusion(('i1', 'j1'), node_data)
        iter_list = extract_dag_creation_components(('i1', 'j1'), dependencies[0], node_data)
        code = generate_dag_generation_code(iter_list, node_data, {'j1': 'm'})
        new_file_name = "dag_extraction.py"
        with open(new_file_name, 'w') as py_file:
            py_file.write(code)
        def wrapper(*args, **kwargs):
            import dag_extraction
            import scheduler
            DAG = dag_extraction.extract_dag(*args, **kwargs)
            num_threads = 8
            schedule = scheduler.schedule_bipartite(DAG, 8)


def generate_executer_code(args, loop_body_1, loop_body_2,it_var1, it_var2):
    code = ""
    code += "def executer({}, scheduler):\n".format(args)
    code += "\tfor s in schedule:\n"
    code += "\t\tfor first, second in s:\n"
    code += "\t\t\tfor {} in first:\n".format(it_var1)
    code += "\t\t\t\t{}\n".format('\n\t\t\t\t'.join(loop_body_1.split('\n')))
    code += "\t\t\tfor {} in second:\n".format(it_var2)
    code += "\t\t\t\t{}\n".format('\n\t\t\t\t'.join(loop_body_2.split('\n')))
    return code

#TODO: test this using function signature.


def spmm_spmm(m, n, Ap, Ai, Ax, B):
    y = np.zeros((m,n))
    z = np.zeros((m,n))
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]

signature = inspect.signature(spmm_spmm)
parameters = signature.parameters
arg_names = [param.name for param in parameters.values()]
print(f"Argument names: {arg_names}")
loop_body_extractor = TileFusionLoopBodyExtractor(('i1', 'j1'))
source = inspect.getsource(spmm_spmm)
tree = ast.parse(source)
loop_body_extractor.visit(tree)
first_loop = loop_body_extractor.first_loop
second_loop = loop_body_extractor.second_loop


# You can also access details about each parameter, such as kind and default value
for name, param in parameters.items():
    print(f"Parameter: {name}, Kind: {param.kind}, Default: {param.default}")

exe = generate_executer_code(", ".join(arg_names), ast.unparse(first_loop), ast.unparse(second_loop), 'i1', 'i2')
print(exe.expandtabs(4))