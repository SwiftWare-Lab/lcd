"""Prepare host values and CUDA device arrays for generated launchers."""

import ast
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from compiler.dag_preprocessing import (
    SyncFreeExecutionData,
    WavefrontExecutionData,
)


class CUDARuntimePreparationError(Exception):
    """Raised when lowered CUDA arguments cannot be prepared."""


class CUDAModuleCompilationError(Exception):
    """Raised when a generated CUDA module cannot be compiled or bound."""


@dataclass
class CompiledCUDAModule:
    """Callable objects and globals belonging to one generated CUDA module."""

    kernel: object
    launcher: object
    namespace: dict[str, object]


@dataclass
class CUDAPreparedArguments:
    """Host/device values ordered for a generated CUDA launcher."""

    launcher_argument_names: tuple[str, ...]
    values: dict[str, object]
    host_values: dict[str, object]
    lifted_variables: tuple[str, ...]

    @property
    def launcher_arguments(self):
        return tuple(
            self.values[name]
            for name in self.launcher_argument_names
        )

    def copy_to_host(self, name):
        """Copy one prepared value back, or return an existing host scalar."""
        if name not in self.values:
            raise CUDARuntimePreparationError(
                f"Unknown prepared CUDA argument {name!r}."
            )
        value = self.values[name]
        copy_method = getattr(value, "copy_to_host", None)
        return copy_method() if copy_method is not None else value

    def copy_lifted_variables_to_host(self):
        """Return lowered local outputs such as ``x`` to host memory."""
        return {
            name: self.copy_to_host(name)
            for name in self.lifted_variables
        }


def compile_cuda_module(
        lowering,
        function_globals=None,
        cuda_module=None,
):
    """Compile generated Python AST and bind its kernel and launcher.

    Numba creates a CUDA dispatcher while this module is executed. Device-code
    specialization remains lazy until the kernel's first launch.
    """
    namespace = dict(function_globals or {})
    module_ast = lowering.module_ast
    if cuda_module is not None:
        namespace["cuda"] = cuda_module
        module_ast = deepcopy(lowering.module_ast)
        module_ast.body = [
            statement
            for statement in module_ast.body
            if not (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "numba"
                and any(
                    alias.name == "cuda"
                    for alias in statement.names
                )
            )
        ]
        module_ast = ast.fix_missing_locations(module_ast)
    try:
        code = compile(
            module_ast,
            filename="<generated-cuda>",
            mode="exec",
        )
        exec(code, namespace, namespace)
    except Exception as error:
        raise CUDAModuleCompilationError(
            "Failed to compile the generated CUDA module."
        ) from error

    try:
        kernel = namespace[lowering.kernel_name]
        launcher = namespace[lowering.launcher_name]
    except KeyError as error:
        raise CUDAModuleCompilationError(
            f"Generated CUDA symbol {error.args[0]!r} was not bound."
        ) from error
    if not callable(kernel) or not callable(launcher):
        raise CUDAModuleCompilationError(
            "Generated CUDA kernel and launcher must both be callable."
        )
    return CompiledCUDAModule(
        kernel=kernel,
        launcher=launcher,
        namespace=namespace,
    )


def prepare_cuda_arguments(
        lowering,
        argument_values,
        execution_data,
        function_globals=None,
        cuda_module=None,
):
    """Create lifted locals and prepare all generated launcher arguments.

    ``argument_values`` contains the original function arguments. Arrays are
    copied to the device, while scalar arguments remain host values. Executor
    arrays are populated from ``execution_data``.
    """
    if cuda_module is None:
        from numba import cuda as cuda_module

    host_values = dict(argument_values)
    host_values.update(
        _evaluate_host_prelude(
            lowering,
            host_values,
            function_globals or {},
        )
    )
    host_values.update(
        _execution_values(lowering.execution_mode, execution_data)
    )

    missing_arguments = [
        name
        for name in lowering.launcher_arguments
        if name not in host_values
    ]
    if missing_arguments:
        raise CUDARuntimePreparationError(
            "Values are missing for generated launcher arguments: "
            + ", ".join(missing_arguments)
        )

    prepared_values = {}
    for name in lowering.launcher_arguments:
        value = host_values[name]
        if name == "wavefront_ptr":
            # The host launcher reads the offsets to issue each kernel launch.
            prepared_values[name] = value
        else:
            prepared_values[name] = _prepare_value(value, cuda_module)

    return CUDAPreparedArguments(
        launcher_argument_names=tuple(lowering.launcher_arguments),
        values=prepared_values,
        host_values=host_values,
        lifted_variables=tuple(lowering.lifted_variables),
    )


def execute_cuda_call(
        compiled_module,
        lowering,
        prepared_arguments,
        function_globals=None,
):
    """Launch prepared CUDA work and restore the original return semantics."""
    compiled_module.launcher(*prepared_arguments.launcher_arguments)
    cuda_module = compiled_module.namespace.get("cuda")
    if cuda_module is None or not hasattr(cuda_module, "synchronize"):
        raise CUDARuntimePreparationError(
            "The compiled CUDA module does not provide cuda.synchronize()."
        )
    cuda_module.synchronize()

    result_values = dict(prepared_arguments.host_values)
    result_values.update(
        prepared_arguments.copy_lifted_variables_to_host()
    )
    return _evaluate_host_epilogue(
        lowering,
        result_values,
        function_globals or {},
    )


def _evaluate_host_prelude(lowering, argument_values, function_globals):
    if not lowering.host_prelude:
        return {}

    namespace = dict(function_globals)
    namespace.update(argument_values)
    prelude = ast.fix_missing_locations(
        ast.Module(
            body=deepcopy(lowering.host_prelude),
            type_ignores=[],
        )
    )
    exec(
        compile(prelude, filename="<cuda-host-prelude>", mode="exec"),
        namespace,
        namespace,
    )

    stored_names = _stored_names(lowering.host_prelude)
    missing_variables = [
        name
        for name in lowering.lifted_variables
        if name not in namespace
    ]
    if missing_variables:
        raise CUDARuntimePreparationError(
            "The removed host prelude did not define lifted variables: "
            + ", ".join(missing_variables)
        )
    return {
        name: namespace[name]
        for name in stored_names
        if name in namespace
    }


def _evaluate_host_epilogue(lowering, result_values, function_globals):
    if not lowering.host_epilogue:
        return None

    namespace = dict(function_globals)
    namespace.update(result_values)
    helper_name = "_generated_cuda_host_epilogue"
    helper = ast.FunctionDef(
        name=helper_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=deepcopy(lowering.host_epilogue),
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[helper], type_ignores=[])
    )
    exec(
        compile(module, filename="<cuda-host-epilogue>", mode="exec"),
        namespace,
        namespace,
    )
    return namespace[helper_name]()


def _stored_names(statements):
    names = []
    for statement in statements:
        for node in ast.walk(statement):
            if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Store)
                    and node.id not in names
            ):
                names.append(node.id)
    return names


def _execution_values(execution_mode, execution_data):
    execution_mode = execution_mode.replace("-", "_")
    if execution_mode == "sync_free":
        if not isinstance(execution_data, SyncFreeExecutionData):
            raise CUDARuntimePreparationError(
                "Sync-free lowering requires SyncFreeExecutionData."
            )
        return {
            "dep_count": execution_data.dep_count,
            "dag_ptr": execution_data.dag_ptr,
            "dag_index": execution_data.dag_index,
        }
    if execution_mode == "wavefront":
        if not isinstance(execution_data, WavefrontExecutionData):
            raise CUDARuntimePreparationError(
                "Wavefront lowering requires WavefrontExecutionData."
            )
        return {
            "wavefronts": execution_data.wavefronts,
            "wavefront_ptr": execution_data.wavefront_ptr,
        }
    return {}


def _prepare_value(value, cuda_module):
    if _is_cuda_array(value) or np.isscalar(value) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return cuda_module.to_device(value)
    raise CUDARuntimePreparationError(
        "CUDA launcher arguments must be scalars, NumPy arrays, sequences, "
        f"or existing CUDA arrays; received {type(value).__name__}."
    )


def _is_cuda_array(value):
    return hasattr(value, "__cuda_array_interface__")
