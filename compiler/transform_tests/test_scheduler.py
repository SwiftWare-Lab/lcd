"""Tests for analytic scheduling features and thread-loop work estimates."""

import ast
import unittest

import numpy as np

from compiler.scheduler import (
    ArchConstants,
    AnalyticScheduler,
    ScheduleSpace,
    UNROLL_FACTORS,
    WAVEFRONT,
    WorkloadFeatures,
    compile_thread_trip_counter,
    extract_workload_features,
    measure_thread_trip_counts,
)


THREAD_LOOP_SOURCE = """
def update(data, indices, indptr):
    n = len(indptr) - 1
    for i in range(n):
        row_end = indptr[i + 1]
        for k_ptr in range(indptr[i], row_end):
            if indices[k_ptr] >= i:
                break
            data[k_ptr] /= data[indptr[indices[k_ptr]]]
            for j_ptr in range(k_ptr + 1, row_end):
                data[j_ptr] -= data[k_ptr]
    return data
"""


class SchedulerTests(unittest.TestCase):
    def test_thread_trip_counter_follows_outer_breaks(self):
        counter = compile_thread_trip_counter(
            ast.parse(THREAD_LOOP_SOURCE),
            "j_ptr",
        )
        data = np.arange(1.0, 7.0)
        original = data.copy()
        counts = measure_thread_trip_counts(
            counter,
            {
                "data": data,
                "indices": np.array([0, 0, 1, 0, 1, 2]),
                "indptr": np.array([0, 1, 3, 6]),
            },
            ("data", "indices", "indptr"),
        )

        self.assertEqual(counts, (1, 2, 1))
        np.testing.assert_array_equal(data, original)

    def test_workload_features_record_thread_loop_work(self):
        features = extract_workload_features(
            dag_successors=[[], [], []],
            wavefront_list=[[0, 1, 2]],
            argument_values={},
            flops_per_iteration=2,
            thread_trip_counts=(1, 2, 1),
        )

        self.assertEqual(features.thread_work_items, 4)
        self.assertEqual(features.thread_loop_invocations, 3)
        self.assertAlmostEqual(features.average_thread_trip_count, 4 / 3)
        self.assertEqual(features.flops, 8.0)

    def test_unroll_selection_uses_average_thread_trip_count(self):
        self.assertEqual(UNROLL_FACTORS, (1, 2, 4, 8))
        scheduler = AnalyticScheduler(
            ArchConstants(
                launch_latency_s=0.0,
                atomic_cost_s=0.0,
                memory_bandwidth_bytes_per_s=1.0e30,
                lane_flops_per_s=1.0,
            )
        )
        space = ScheduleSpace.from_options(
            execution_mode=WAVEFRONT,
            threads_per_block=128,
            thread_mapped=True,
            n_warp=1,
        )

        short = WorkloadFeatures(
            rows=10,
            edges=0,
            levels=1,
            nonzeros=160,
            flops=160.0,
            memory_bytes=0,
            thread_work_items=160,
            thread_loop_invocations=10,
        )
        long = WorkloadFeatures(
            rows=10,
            edges=0,
            levels=1,
            nonzeros=480,
            flops=480.0,
            memory_bytes=0,
            thread_work_items=480,
            thread_loop_invocations=10,
        )

        self.assertEqual(scheduler.schedule(short, space).thread_unroll, 1)
        self.assertEqual(scheduler.schedule(long, space).thread_unroll, 2)


if __name__ == "__main__":
    unittest.main()
