import inspect
import functools


import types
import ast
from compiler.dag_codegen import (
    compile_dag_builder,
    create_dag_plan,
    generate_dag_source,
)
from compiler.cuda_lowering import lower_to_numba_cuda
from compiler.cuda_runtime import (
    compile_cuda_module,
    execute_cuda_call,
    prepare_cuda_arguments,
)
from compiler.dag_preprocessing import preprocess_dag_output
from compiler.legality_check import finalize_legality_of_fusion, transform_ast, generate_dag_generation_code, \
    extract_dependences
from compiler.legality_check import extract_node_data, extract_dag_creation_components
from compiler.spf_raising import SPFBuilder, SparseFusionTransformer
import os
from compiler.spf_raising import LoopNode


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
                constraints, dependence_analysis = legality_res
                dependencies = dependence_analysis.ordering_dependences
                if not dependencies:
                    transformed_tree = transform_ast(transformation, cur_ast)
                    transformed_tree = ast.fix_missing_locations(transformed_tree)
                    cur_ast = transformed_tree
                    continue
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
        constraints, dependence_analysis = finalize_legality_of_fusion(
            (loop, loop), cur_ast, function_props
        )
        print('\n-----------------------------------------------\n'.join(
            map(str, dependence_analysis.ordering_dependences)
        ))
        print("Reduction dependences:")
        print('\n-----------------------------------------------\n'.join(
            map(str, dependence_analysis.reduction_dependences)
        ))

        signature = inspect.signature(function)
        argument_names = list(signature.parameters)
        spf_builder = SPFBuilder()
        root_node = spf_builder.visit(cur_ast)
        node_data = extract_node_data(root_node)
        dag_plan = create_dag_plan(
            dependence_analysis.ordering_dependences,
            node_data,
            parallel_iterator=loop,
            arguments=argument_names,
        )
        validate_endpoints = kwargs.get("validate_dag_endpoints", False)
        execution_mode = kwargs.get("execution_mode", "sync_free")
        dag_source = generate_dag_source(
            dag_plan,
            validate_endpoints=validate_endpoints,
            execution_mode=execution_mode,
        )
        dag_builder = compile_dag_builder(
            dag_plan,
            validate_endpoints=validate_endpoints,
            execution_mode=execution_mode,
        )
        cuda_execution_mode = (
            "sync_free"
            if execution_mode.replace("-", "_") == "sync_free"
            else "wavefront"
        )
        cuda_lowering = lower_to_numba_cuda(
            cur_ast,
            loop_iterator=loop,
            reductions=dependence_analysis,
            execution_mode=cuda_execution_mode,
            sync_free_ready_count=kwargs.get(
                "sync_free_ready_count",
                0,
            ),
            wavefront_threads_per_block=kwargs.get(
                "threads_per_block",
                128,
            ),
        )
        compiled_cuda_module = compile_cuda_module(
            cuda_lowering,
            function_globals=function.__globals__,
        )

        def build_dag(*args, **call_kwargs):
            bound_arguments = signature.bind(*args, **call_kwargs)
            bound_arguments.apply_defaults()
            return dag_builder(*[
                bound_arguments.arguments[name]
                for name in argument_names
            ])

        def build_execution_data(*args, **call_kwargs):
            dag_successors, scheduling_data = build_dag(
                *args,
                **call_kwargs,
            )
            return preprocess_dag_output(
                dag_successors,
                scheduling_data,
                execution_mode,
            )

        def prepare_cuda_call(*args, **call_kwargs):
            bound_arguments = signature.bind(*args, **call_kwargs)
            bound_arguments.apply_defaults()
            execution_data = build_execution_data(
                *args,
                **call_kwargs,
            )
            return prepare_cuda_arguments(
                lowering=cuda_lowering,
                argument_values=bound_arguments.arguments,
                execution_data=execution_data,
                function_globals=function.__globals__,
                cuda_module=compiled_cuda_module.namespace["cuda"],
            )

        @functools.wraps(function)
        def wrapper(*args, **call_kwargs):
            prepared_arguments = prepare_cuda_call(
                *args,
                **call_kwargs,
            )
            return execute_cuda_call(
                compiled_module=compiled_cuda_module,
                lowering=cuda_lowering,
                prepared_arguments=prepared_arguments,
                function_globals=function.__globals__,
            )

        # Keep code-generation artifacts on the decorated function until the
        # executor/scheduler lowering stage consumes them.
        wrapper.original_function = function
        wrapper.build_dag = build_dag
        wrapper.build_execution_data = build_execution_data
        wrapper.prepare_cuda_arguments = prepare_cuda_call
        wrapper.dag_plan = dag_plan
        wrapper.dag_source = dag_source
        wrapper.execution_mode = execution_mode
        wrapper.dependence_analysis = dependence_analysis
        wrapper.reduction_statements = dependence_analysis.atomic_statements
        wrapper.cuda_lowering = cuda_lowering
        wrapper.compiled_cuda_module = compiled_cuda_module
        wrapper.cuda_kernel = compiled_cuda_module.kernel
        wrapper.cuda_launcher = compiled_cuda_module.launcher
        wrapper.cuda_module_ast = cuda_lowering.module_ast
        wrapper.cuda_kernel_ast = cuda_lowering.kernel_ast
        wrapper.cuda_source = cuda_lowering.source
        wrapper.cuda_kernel_source = ast.unparse(
            cuda_lowering.kernel_ast
        )
        wrapper.cuda_launcher_ast = cuda_lowering.launcher_ast
        wrapper.cuda_launcher_source = ast.unparse(
            cuda_lowering.launcher_ast
        )
        return wrapper
    return decorator
