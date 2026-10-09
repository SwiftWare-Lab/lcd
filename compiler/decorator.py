import inspect
import functools


import types
import ast
from compiler.ast_normalization import normalize_aliases
from compiler.dag_codegen import (
    COUNTER_DEPENDENCE,
    PUSH_DEPENDENCE,
    compile_dag_builder,
    create_dag_plan,
    direct_dependence,
    generate_dag_ast,
    generate_dag_source,
)
from compiler.cuda_lowering import lower_to_numba_cuda
from compiler.cuda_runtime import (
    compile_cuda_module,
    evaluate_host_prelude,
    execute_cuda_call,
    prepare_cuda_arguments,
)
from compiler.dag_preprocessing import (
    preprocess_dag_output,
    preprocess_sync_free,
    preprocess_warp_tasks,
    preprocess_wavefront,
    wavefronts_from_dag,
)
from compiler.scheduler import (
    AUTO,
    SYNC_FREE,
    AnalyticScheduler,
    SchedulePlan,
    ScheduleSpace,
    count_inner_loop_flops,
    extract_workload_features,
    inspection_fingerprint,
    pointer_array_names,
)
from compiler.legality_check import finalize_legality_of_fusion, transform_ast, generate_dag_generation_code, \
    extract_dependences
from compiler.legality_check import extract_node_data, extract_dag_creation_components
from compiler.spf_raising import build_spf_from_ast, SparseFusionTransformer
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
                root_node = build_spf_from_ast(cur_ast)
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
        loop = kwargs.get(
            'loop',
            kwargs.get('block_map', kwargs.get('block_loop', None)),
        )
        if loop is None:
            raise ValueError(
                "parallel requires a block-mapped iterator via `loop`, "
                "`block_map`, or `block_loop`."
            )
        function_props = kwargs.get('function_props', {})
        num_threads = kwargs.get('num_threads', 8)
        source = inspect.getsource(function)
        source = '\n'.join([l for l in source.splitlines() if not l.startswith("@")])
        print(source)
        old_code_obj = function.__code__
        cur_ast = normalize_aliases(ast.parse(source))
        constraints, dependence_analysis = finalize_legality_of_fusion(
            (loop, loop),
            cur_ast,
            function_props,
            block_iterator=loop,
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
        root_node = build_spf_from_ast(
            cur_ast,
            block_iterator=loop,
        )
        node_data = extract_node_data(root_node)
        execution_mode = kwargs.get("execution_mode", AUTO)
        thread_loop = kwargs.get(
            "thread_loop",
            kwargs.get("thread_map", None),
        )
        thread_unroll = kwargs.get(
            "thread_unroll",
            kwargs.get("thread_unroll_factor", None),
        )
        space_options = dict(
            execution_mode=execution_mode,
            threads_per_block=kwargs.get("threads_per_block"),
            thread_unroll=thread_unroll,
            thread_mapped=thread_loop is not None,
            n_warp=kwargs.get("n_warp"),
        )
        band_width = kwargs.get("band_width", 1)
        scheduler = kwargs.get("scheduler") or AnalyticScheduler(
            kwargs.get("arch_constants"),
        )
        lowered_variants = {}
        compiled_variants = {}
        dependence = None

        def lower_variant(lowering_key):
            if lowering_key not in lowered_variants:
                (
                    cuda_execution_mode,
                    threads_per_block,
                    unroll_factor,
                    n_warp,
                    dependence_protocol,
                ) = lowering_key
                lowered_variants[lowering_key] = lower_to_numba_cuda(
                    cur_ast,
                    loop_iterator=loop,
                    reductions=dependence_analysis,
                    execution_mode=cuda_execution_mode,
                    sync_free_ready_count=kwargs.get(
                        "sync_free_ready_count",
                        0,
                    ),
                    wavefront_threads_per_block=threads_per_block,
                    thread_loop_iterator=thread_loop,
                    thread_unroll_factor=unroll_factor,
                    n_warp=n_warp,
                    dependence_protocol=dependence_protocol,
                    direct_dependence=(
                        None
                        if dependence_protocol == COUNTER_DEPENDENCE
                        else dependence
                    ),
                )
            return lowered_variants[lowering_key]

        def compile_variant(lowering_key):
            cuda_lowering = lower_variant(lowering_key)
            if lowering_key not in compiled_variants:
                compiled_variants[lowering_key] = compile_cuda_module(
                    cuda_lowering,
                    function_globals=function.__globals__,
                )
            return cuda_lowering, compiled_variants[lowering_key]

        host_lowering = lower_variant(
            ScheduleSpace.from_options(**space_options).configurations()[0]
        )
        dag_argument_names = argument_names + [
            name
            for name in host_lowering.lifted_loop_parameters
            if name not in argument_names
        ]
        dag_plan = create_dag_plan(
            dependence_analysis.ordering_dependences,
            node_data,
            parallel_iterator=loop,
            arguments=dag_argument_names,
            function_props=function_props,
        )
        validate_endpoints = kwargs.get("validate_dag_endpoints", False)
        dependence = direct_dependence(
            dag_plan,
            thread_loop,
            validate_endpoints=validate_endpoints,
        )
        schedule_space = ScheduleSpace.from_options(
            **space_options,
            direct_protocol=None if dependence is None else dependence.kind,
            dependence_protocol=kwargs.get("dependence_protocol"),
        )
        fixed_schedule = (
            schedule_space.fixed_schedule()
            if schedule_space.is_fixed
            else None
        )
        edge_multiplicity = any(
            configuration[-1] == PUSH_DEPENDENCE
            for configuration in schedule_space.configurations()
        )
        inspector_mode = (
            execution_mode
            if fixed_schedule is not None
            else SYNC_FREE
        )
        inspector_options = dict(
            validate_endpoints=validate_endpoints,
            execution_mode=inspector_mode,
            edge_multiplicity=edge_multiplicity,
        )
        dag_source = generate_dag_source(dag_plan, **inspector_options)
        dag_builder = compile_dag_builder(dag_plan, **inspector_options)
        inspector_names = {
            node.id
            for node in ast.walk(
                generate_dag_ast(dag_plan, **inspector_options)
            )
            if isinstance(node, ast.Name)
        }
        fingerprint_names = [
            name
            for name in dag_argument_names
            if name in inspector_names
        ] + pointer_array_names(function_props)
        flops_per_iteration = count_inner_loop_flops(
            cur_ast,
            loop,
            thread_loop,
        )
        schedule_plans = {}

        def bind_arguments(args, call_kwargs):
            bound_arguments = signature.bind(*args, **call_kwargs)
            bound_arguments.apply_defaults()
            return bound_arguments.arguments

        def dag_values_for(argument_values):
            dag_values = dict(argument_values)
            dag_values.update(
                evaluate_host_prelude(
                    lowering=host_lowering,
                    argument_values=dag_values,
                    function_globals=function.__globals__,
                )
            )
            return dag_values

        def run_inspector(dag_values):
            return dag_builder(*[
                dag_values[name]
                for name in dag_argument_names
            ])

        def build_dag(*args, **call_kwargs):
            return run_inspector(
                dag_values_for(bind_arguments(args, call_kwargs))
            )[:2]

        def warp_mapped(schedule):
            return lower_variant(schedule.lowering_key).warp_mapped

        def warp_task_data(schedule, dag_successors, wavefront_list, edge_count):
            pushed = (
                lower_variant(schedule.lowering_key).dependence_protocol
                == PUSH_DEPENDENCE
            )
            return preprocess_warp_tasks(
                dag_successors,
                wavefront_list,
                band_width,
                edge_count=edge_count if pushed else None,
            )

        def create_schedule_plan(dag_values):
            dag_successors, scheduling_data, *edge_count = run_inspector(
                dag_values,
            )
            edge_count = edge_count[0] if edge_count else None
            if fixed_schedule is not None:
                if not warp_mapped(fixed_schedule):
                    return SchedulePlan(
                        schedule=fixed_schedule,
                        execution_data=preprocess_dag_output(
                            dag_successors,
                            scheduling_data,
                            execution_mode,
                        ),
                    )
                wavefront_list = (
                    scheduling_data
                    if fixed_schedule.execution_mode != SYNC_FREE
                    else wavefronts_from_dag(dag_successors, scheduling_data)
                )
                return SchedulePlan(
                    schedule=fixed_schedule,
                    execution_data=warp_task_data(
                        fixed_schedule,
                        dag_successors,
                        wavefront_list,
                        edge_count,
                    ),
                )
            wavefront_list = wavefronts_from_dag(
                dag_successors,
                scheduling_data,
            )
            features = extract_workload_features(
                dag_successors,
                wavefront_list,
                dag_values,
                function_props=function_props,
                flops_per_iteration=flops_per_iteration,
            )
            schedule = scheduler.schedule(features, schedule_space)
            if warp_mapped(schedule):
                execution_data = warp_task_data(
                    schedule,
                    dag_successors,
                    wavefront_list,
                    edge_count,
                )
            elif schedule.execution_mode == SYNC_FREE:
                execution_data = preprocess_sync_free(
                    dag_successors,
                    scheduling_data,
                )
            else:
                execution_data = preprocess_wavefront(
                    dag_successors,
                    wavefront_list,
                )
            return SchedulePlan(
                schedule=schedule,
                execution_data=execution_data,
                features=features,
            )

        def plan_for(argument_values):
            dag_values = dag_values_for(argument_values)
            fingerprint = inspection_fingerprint(
                dag_values,
                fingerprint_names,
            )
            plan = schedule_plans.get(fingerprint)
            if plan is None:
                plan = create_schedule_plan(dag_values)
                if len(schedule_plans) >= SCHEDULE_CACHE_SIZE:
                    schedule_plans.pop(next(iter(schedule_plans)))
                schedule_plans[fingerprint] = plan
            return plan

        def schedule_plan(*args, **call_kwargs):
            return plan_for(bind_arguments(args, call_kwargs))

        def schedule_call(*args, **call_kwargs):
            return schedule_plan(*args, **call_kwargs).schedule

        def build_execution_data(*args, **call_kwargs):
            return schedule_plan(*args, **call_kwargs).execution_data

        def publish_variant(schedule, cuda_lowering, compiled_cuda_module):
            if wrapper.current_schedule == schedule:
                return
            wrapper.current_schedule = schedule
            wrapper.__dict__.update(
                _variant_attributes(cuda_lowering, compiled_cuda_module)
            )

        def resolve_call(args, call_kwargs):
            argument_values = bind_arguments(args, call_kwargs)
            plan = plan_for(argument_values)
            cuda_lowering, compiled_cuda_module = compile_variant(
                plan.schedule.lowering_key
            )
            publish_variant(plan.schedule, cuda_lowering, compiled_cuda_module)
            prepared_arguments = prepare_cuda_arguments(
                lowering=cuda_lowering,
                argument_values=argument_values,
                execution_data=plan.execution_data,
                function_globals=function.__globals__,
                cuda_module=compiled_cuda_module.namespace["cuda"],
            )
            return cuda_lowering, compiled_cuda_module, prepared_arguments

        def prepare_cuda_call(*args, **call_kwargs):
            return resolve_call(args, call_kwargs)[2]

        @functools.wraps(function)
        def wrapper(*args, **call_kwargs):
            cuda_lowering, compiled_cuda_module, prepared_arguments = (
                resolve_call(args, call_kwargs)
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
        wrapper.schedule_plan = schedule_plan
        wrapper.schedule_call = schedule_call
        wrapper.schedule_space = schedule_space
        wrapper.scheduler = scheduler
        wrapper.dag_plan = dag_plan
        wrapper.dag_source = dag_source
        wrapper.execution_mode = execution_mode
        wrapper.dependence_analysis = dependence_analysis
        wrapper.reduction_statements = dependence_analysis.atomic_statements
        wrapper.current_schedule = None
        wrapper.__dict__.update(dict.fromkeys(VARIANT_ATTRIBUTES))
        if fixed_schedule is not None:
            publish_variant(
                fixed_schedule,
                *compile_variant(fixed_schedule.lowering_key),
            )
        return wrapper
    return decorator


SCHEDULE_CACHE_SIZE = 8
VARIANT_ATTRIBUTES = (
    "cuda_lowering",
    "compiled_cuda_module",
    "cuda_kernel",
    "cuda_launcher",
    "cuda_module_ast",
    "cuda_kernel_ast",
    "cuda_source",
    "cuda_kernel_source",
    "cuda_launcher_ast",
    "cuda_launcher_source",
)


def _variant_attributes(cuda_lowering, compiled_cuda_module):
    return {
        "cuda_lowering": cuda_lowering,
        "compiled_cuda_module": compiled_cuda_module,
        "cuda_kernel": compiled_cuda_module.kernel,
        "cuda_launcher": compiled_cuda_module.launcher,
        "cuda_module_ast": cuda_lowering.module_ast,
        "cuda_kernel_ast": cuda_lowering.kernel_ast,
        "cuda_source": cuda_lowering.source,
        "cuda_kernel_source": ast.unparse(cuda_lowering.kernel_ast),
        "cuda_launcher_ast": cuda_lowering.launcher_ast,
        "cuda_launcher_source": ast.unparse(cuda_lowering.launcher_ast),
    }
