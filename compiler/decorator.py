import inspect


import types
import ast
from compiler.legality_check import finalize_legality_of_fusion, transform_ast, generate_dag_generation_code, \
    extract_dependences
from compiler.legality_check import extract_node_data, extract_dag_creation_components
from compiler.spf_raising import SPFBuilder, SparseFusionTransformer
import os
from compiler.spf_def import LoopNode


#TODO: Wierd design, need to think about how to fix this.
def fusion(**kwargs):
    def decorator(function: types.FunctionType):
        transformations = kwargs['transformation']
        function_props = kwargs.get('function_props', {})
        num_threads = kwargs.get('num_threads', 8)
        source = inspect.getsource(function)
        source = '\n'.join([l for l in source.splitlines() if not l.startswith("@")])

        old_code_obj = function.__code__
        cur_ast = ast.parse(source)
        for transformation in transformations:
            legality_res = finalize_legality_of_fusion(transformation, cur_ast, function_props)
            if isinstance(legality_res, bool) and legality_res:
                transformed_tree = transform_ast(transformation, cur_ast)
                print(ast.unparse(transformed_tree))
                transformed_tree = ast.fix_missing_locations(transformed_tree)
                cur_ast = transformed_tree
            else:
                constraints, dependencies = legality_res
                signature = inspect.signature(function)
                parameters = signature.parameters
                arg_names = [param.name for param in parameters.values()]
                spf_builder = SPFBuilder()
                root_node = spf_builder.visit(cur_ast)
                node_data = extract_node_data(root_node)
                iter_list = extract_dag_creation_components(transformation, dependencies[0], node_data)
                dag_code = generate_dag_generation_code(arg_names, iter_list, node_data, {'j1': 'm'})
                sp_fusion_transformer = SparseFusionTransformer(transformation)
                new_ast = sp_fusion_transformer.visit(cur_ast)
                new_ast = ast.fix_missing_locations(cur_ast)
                executer_code = ast.unparse(new_ast)
                print(dag_code.expandtabs(4))
                print(executer_code)
                # get current directory

                current_dir = os.path.dirname(os.path.abspath(__file__))
                # create a new file in the compiler directory
                new_file_name = os.path.join(current_dir, "generated_code.py")
                with open(new_file_name, 'w') as py_file:
                    py_file.write(dag_code)
                    # py_file.write(executer_code)
                def wrapper(*args, **kwargs):
                    from compiler import generated_code
                    from compiler import scheduler
                    DAG = generated_code.extract_dag(*args, **kwargs)
                    schedule = scheduler.schedule_bipartite(DAG, num_threads)
                    new_code_obj = compile(new_ast, old_code_obj.co_filename, 'exec')
                    new_f = types.FunctionType(new_code_obj.co_consts[0], function.__globals__)
                    output = new_f(*args, schedule)
                    return output
                return wrapper

        new_code_obj = compile(transformed_tree, old_code_obj.co_filename, 'exec')
        new_f = types.FunctionType(new_code_obj.co_consts[0], function.__globals__)
        return new_f
    return decorator

def generate_executer_code(arg_names, loop_body_1, loop_body_2, ret_stmnt, transformation):
    it_var1 = transformation[0]
    it_var2 = transformation[1]
    args = ", ".join(arg_names)
    code = ""
    code += "def executer({}, schedule):\n".format(args)
    code += "\tfor s in schedule:\n"
    code += "\t\tfor first, second in s:\n"
    code += "\t\t\tfor {} in first:\n".format(it_var1)
    code += "\t\t\t\t{}\n".format('\n\t\t\t\t'.join(loop_body_1.split('\n')))
    code += "\t\t\tfor {} in second:\n".format(it_var2)
    code += "\t\t\t\t{}\n".format('\n\t\t\t\t'.join(loop_body_2.split('\n')))
    code += "\t" + ret_stmnt + "\n"

    return code


def parallel(**kwargs):
    def decorator(function: types.FunctionType):
        loop = kwargs['loop']
        function_props = kwargs.get('function_props', {})
        num_threads = kwargs.get('num_threads', 8)
        source = inspect.getsource(function)
        source = '\n'.join([l for l in source.splitlines() if not l.startswith("@")])
        print(source)
        old_code_obj = function.__code__
        cur_ast = ast.parse(source)
        finalize_legality_of_fusion((loop, loop), cur_ast, function_props)
        # spf_builder = SPFBuilder()
        # root_node = spf_builder.visit(cur_ast)
        # node_data = extract_node_data(root_node)
        # loop_bounds = {}
        # mapping_nodes = {}
        # mapping = (loop, loop)
        # for lid, lnode in node_data.items():
        #     if isinstance(lnode, LoopNode):
        #         loop_bounds[lnode.iter_var] = lnode.bound
        #         if lnode.iter_var in mapping:
        #             mapping_nodes[lnode.iter_var] = lnode
        # dependences = extract_dependences(node_data, mapping, mapping_nodes)
    return decorator