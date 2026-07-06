"""Compact generated DAG schedules into CUDA executor data structures."""

from dataclasses import dataclass

import numpy as np


class DAGPreprocessingError(Exception):
    """Raised when generated DAG data cannot form executor inputs."""


@dataclass(frozen=True)
class SyncFreeExecutionData:
    """CSR successors and mutable incoming-dependency counts."""

    dep_count: np.ndarray
    dag_ptr: np.ndarray
    dag_index: np.ndarray

    @property
    def launcher_arguments(self):
        """Arguments in the order expected by the sync-free launcher."""
        return self.dep_count, self.dag_ptr, self.dag_index


@dataclass(frozen=True)
class WavefrontExecutionData:
    """Flattened wavefront nodes and offsets for each wavefront."""

    wavefronts: np.ndarray
    wavefront_ptr: np.ndarray

    @property
    def launcher_arguments(self):
        """Arguments in the order expected by the wavefront launcher."""
        return self.wavefronts, self.wavefront_ptr


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
