"""Compact generated DAG schedules into CUDA executor data structures."""

from dataclasses import dataclass, field, replace

import numpy as np


class DAGPreprocessingError(Exception):
    """Raised when generated DAG data cannot form executor inputs."""


@dataclass(frozen=True)
class SyncFreeExecutionData:
    """CSR successors and mutable incoming-dependency counts."""

    dep_count: np.ndarray
    dag_ptr: np.ndarray
    dag_index: np.ndarray
    initial_dep_count: np.ndarray | None = None
    auxiliary_values: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def launcher_arguments(self):
        """Arguments in the order expected by the sync-free launcher."""
        return self.dep_count, self.dag_ptr, self.dag_index


@dataclass(frozen=True)
class WavefrontExecutionData:
    """Flattened wavefront nodes and offsets for each wavefront."""

    wavefronts: np.ndarray
    wavefront_ptr: np.ndarray
    auxiliary_values: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def launcher_arguments(self):
        """Arguments in the order expected by the wavefront launcher."""
        return self.wavefronts, self.wavefront_ptr


@dataclass(frozen=True)
class WarpTaskExecutionData:
    task_nodes: np.ndarray
    task_ptr: np.ndarray
    band_ptr: np.ndarray
    dep_count: np.ndarray
    dag_ptr: np.ndarray
    dag_index: np.ndarray
    initial_dep_count: np.ndarray | None = None
    auxiliary_values: dict[str, np.ndarray] = field(default_factory=dict)


def preprocess_dag_output(
        dag_successors,
        scheduling_data,
        execution_mode,
):
    """Convert ``extract_dag`` output to the selected executor format."""
    execution_mode = execution_mode.replace("-", "_")
    if execution_mode == "sync_free":
        return preprocess_sync_free(dag_successors, scheduling_data)
    if execution_mode == "wavefront":
        return preprocess_wavefront(dag_successors, scheduling_data)
    raise DAGPreprocessingError(
        f"Unknown execution mode {execution_mode!r}; expected sync_free "
        "or wavefront."
    )


def attach_auxiliary_values(execution_data, auxiliary_values):
    """Attach immutable inspector outputs to any supported executor format."""
    if not auxiliary_values:
        return execution_data
    if not isinstance(
            execution_data,
            (SyncFreeExecutionData, WavefrontExecutionData,
             WarpTaskExecutionData),
    ):
        raise DAGPreprocessingError(
            "Auxiliary values require recognized DAG execution data."
        )
    collisions = set(execution_data.auxiliary_values) & set(auxiliary_values)
    if collisions:
        raise DAGPreprocessingError(
            "Duplicate auxiliary execution values: "
            + ", ".join(sorted(collisions))
        )
    combined = dict(execution_data.auxiliary_values)
    combined.update(auxiliary_values)
    return replace(execution_data, auxiliary_values=combined)


def preprocess_sync_free(dag_successors, dep_count):
    """Build CSR out-edge arrays used to release successor tasks."""
    dag_ptr, dag_index = compact_successors(dag_successors)
    dep_count = np.asarray(dep_count, dtype=np.int32)
    if dep_count.ndim != 1 or len(dep_count) != len(dag_successors):
        raise DAGPreprocessingError(
            "dep_count must contain one value for every DAG node."
        )
    if np.any(dep_count < 0):
        raise DAGPreprocessingError(
            "Dependency counts cannot be negative."
        )
    return SyncFreeExecutionData(
        # The kernel mutates this data, so preserve the generated counts.
        dep_count=dep_count.copy(),
        initial_dep_count=dep_count.copy(),
        dag_ptr=dag_ptr,
        dag_index=dag_index,
    )


def compact_successors(dag_successors):
    """Convert successor lists to CSR ``dag_ptr`` and ``dag_index``."""
    node_count = len(dag_successors)
    dag_ptr = np.zeros(node_count + 1, dtype=np.int32)
    dag_index_values = []

    for source, successors in enumerate(dag_successors):
        try:
            unique_successors = sorted({
                int(destination)
                for destination in successors
            })
        except (TypeError, ValueError) as error:
            raise DAGPreprocessingError(
                f"Invalid successor list for node {source}."
            ) from error

        for destination in unique_successors:
            if not 0 <= destination < node_count:
                raise DAGPreprocessingError(
                    f"DAG edge {source} -> {destination} is out of bounds."
                )
        dag_index_values.extend(unique_successors)
        dag_ptr[source + 1] = len(dag_index_values)

    return (
        dag_ptr,
        np.asarray(dag_index_values, dtype=np.int32),
    )


def wavefronts_from_dag(dag_successors, dep_count):
    remaining_dep_count = list(dep_count)
    wavefront_list = []
    wavefront = [
        node
        for node, count in enumerate(remaining_dep_count)
        if count == 0
    ]
    while wavefront:
        wavefront_list.append(wavefront)
        new_wavefront = set()
        for node in wavefront:
            for successor in dag_successors[node]:
                remaining_dep_count[successor] -= 1
                if remaining_dep_count[successor] == 0:
                    new_wavefront.add(successor)
        wavefront = sorted(new_wavefront)
    return wavefront_list


def preprocess_wavefront(dag_successors, wavefront_list):
    """Flatten wavefronts and build the offset array used by the launcher."""
    node_count = len(dag_successors)
    wavefront_values = []
    wavefront_ptr = np.zeros(len(wavefront_list) + 1, dtype=np.int32)
    scheduled_nodes = set()

    for wave_number, wavefront in enumerate(wavefront_list):
        try:
            nodes = [int(node) for node in wavefront]
        except (TypeError, ValueError) as error:
            raise DAGPreprocessingError(
                f"Invalid node in wavefront {wave_number}."
            ) from error

        for node in nodes:
            if not 0 <= node < node_count:
                raise DAGPreprocessingError(
                    f"Wavefront node {node} is out of bounds."
                )
            if node in scheduled_nodes:
                raise DAGPreprocessingError(
                    f"DAG node {node} occurs in more than one wavefront."
                )
            scheduled_nodes.add(node)
        wavefront_values.extend(nodes)
        wavefront_ptr[wave_number + 1] = len(wavefront_values)

    if len(scheduled_nodes) != node_count:
        missing = sorted(set(range(node_count)) - scheduled_nodes)
        raise DAGPreprocessingError(
            "Wavefront scheduling did not cover every DAG node; the DAG may "
            f"contain a cycle. Missing nodes: {missing}."
        )

    return WavefrontExecutionData(
        wavefronts=np.asarray(wavefront_values, dtype=np.int32),
        wavefront_ptr=wavefront_ptr,
    )


def preprocess_warp_tasks(
        dag_successors,
        wavefront_list,
        band_width=1,
        edge_count=None,
):
    if not isinstance(band_width, int) or band_width <= 0:
        raise DAGPreprocessingError(
            "The band width must be a positive integer."
        )
    node_count = len(dag_successors)
    dag_ptr, dag_index = compact_successors(dag_successors)
    wavefront_data = preprocess_wavefront(dag_successors, wavefront_list)
    nodes = wavefront_data.wavefronts
    level_ptr = wavefront_data.wavefront_ptr
    if edge_count is None:
        dep_count = np.bincount(
            dag_index,
            minlength=node_count,
        ).astype(np.int32)
    else:
        dep_count = np.asarray(edge_count, dtype=np.int32)
        if dep_count.shape != (node_count,):
            raise DAGPreprocessingError(
                "edge_count must contain one value for every DAG node."
            )

    if band_width == 1:
        task_nodes = nodes
        task_ptr = np.arange(node_count + 1, dtype=np.int32)
        band_ptr = level_ptr.copy()
    else:
        task_nodes, task_ptr, band_ptr = _band_chains(
            nodes,
            level_ptr,
            dag_ptr,
            dag_index,
            band_width,
        )

    return WarpTaskExecutionData(
        task_nodes=task_nodes,
        task_ptr=task_ptr,
        band_ptr=band_ptr,
        dep_count=dep_count.copy(),
        initial_dep_count=dep_count.copy(),
        dag_ptr=dag_ptr,
        dag_index=dag_index,
    )


def _band_chains(nodes, level_ptr, dag_ptr, dag_index, band_width):
    level_count = len(level_ptr) - 1
    task_nodes = []
    task_ptr = [0]
    band_ptr = [0]
    for first_level in range(0, level_count, band_width):
        last_level = min(first_level + band_width, level_count)
        band = nodes[level_ptr[first_level]:level_ptr[last_level]].tolist()
        local = {node: position for position, node in enumerate(band)}
        parent = list(range(len(band)))

        def find(position):
            root = position
            while parent[root] != root:
                root = parent[root]
            while parent[position] != root:
                parent[position], position = root, parent[position]
            return root

        for position, node in enumerate(band):
            for successor in dag_index[dag_ptr[node]:dag_ptr[node + 1]]:
                successor_position = local.get(int(successor))
                if successor_position is not None:
                    root = find(position)
                    successor_root = find(successor_position)
                    if root != successor_root:
                        parent[successor_root] = root

        chains = {}
        for position, node in enumerate(band):
            chains.setdefault(find(position), []).append(node)
        for chain in chains.values():
            task_nodes.extend(chain)
            task_ptr.append(len(task_nodes))
        band_ptr.append(len(task_ptr) - 1)

    return (
        np.asarray(task_nodes, dtype=np.int32),
        np.asarray(task_ptr, dtype=np.int32),
        np.asarray(band_ptr, dtype=np.int32),
    )
