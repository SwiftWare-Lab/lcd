from numba.core import ir_utils, config
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import *
from numba.core.untyped_passes import IRProcessing
from numba.core.analysis import compute_live_map, compute_dead_maps, compute_live_variables

from matplotlib import pyplot as plt
import time

from analysis import extract_data_value, unroll_simple_loops

import numba


config.DEBUG_ARRAY_OPT = 3


@register_pass(mutates_CFG=True, analysis_only=True)
class LoopFusion1(FunctionPass):
    _name = "loop_fusion1"

    def __init__(self):
        super().__init__(self)


    def run_pass(self, state):

        # state contains the FunctionIR to be mutated,
        mutate = False
        # a very nasty way to fuse two adjacent loops. Not generic, just for testing
        func_ir = state.func_ir
        blocks = func_ir.blocks
        dprint_func_ir(func_ir, "before maximize fusion down")
        cfg = ir_utils.compute_cfg_from_blocks(blocks)
        use_defs = compute_use_defs(func_ir.blocks)
        # plot cfg
        graphiz_digraph = cfg.render_dot("diag_cfg.dot")
        # convert graphvizdigraph to pdf
        graphiz_digraph.format = "pdf"
        #graphiz_digraph.view()
        cfg.dump()

        live_map = compute_live_map(cfg, func_ir.blocks, use_defs.usemap, use_defs.defmap)
        dead_map = compute_dead_maps(cfg, func_ir.blocks, live_map, use_defs.defmap)
        live_var = compute_live_variables(cfg, func_ir.blocks, use_defs.defmap, dead_map.combined)

        raw_dependencies = {}
        for block_label, live_vars in live_map.items():
            for var in live_vars:
                if var in use_defs.defmap[block_label]:
                    if block_label not in raw_dependencies:
                        raw_dependencies[block_label] = []
                    raw_dependencies[block_label].append(var)

        print("Live Map:", raw_dependencies)




        # dot.render("cfg_diagram", view=True)

        order = find_topo_order(blocks)
        return mutate  # the pass has not mutated the IR


# mutate_CFG since we add a new loop for remaining iterations
# not analysis_only since unrolling changes IR
@register_pass(mutates_CFG=True, analysis_only=False)
class LoopUnrolling(FunctionPass):
    """ Implements loop unrolling on loops with only 1 basic block in their body """
    _name = "loop_unrolling"

    def __init__(self):
        super().__init__(self)

    def run_pass(self, state):
        mutate = False

        func_ir = state.func_ir
        blocks = func_ir.blocks
        dprint_func_ir(func_ir, "before maximize fusion down")
        cfg = ir_utils.compute_cfg_from_blocks(blocks)
        # cfg.dump()

        data_values, _ = extract_data_value(blocks)
        mutate = unroll_simple_loops(cfg, blocks, data_values, factor=2)
        cfg = ir_utils.compute_cfg_from_blocks(blocks)
        cfg.dump()

        return mutate


class MyCompiler(CompilerBase):  # custom compiler extends from CompilerBase

    def define_pipelines(self):
        # define a new set of pipelines (just one in this case) and for ease
        # base it on an existing pipeline from the DefaultPassBuilder,
        # namely the "nopython" pipeline
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        # Add the new pass to run after IRProcessing
        pm.add_pass_after(LoopUnrolling, IRProcessing)
        # pm.add_pass_after(LoopFusion1, IRProcessing)
        #pm.add_pass_after(ParFdd, IRProcessing)
        # finalize
        pm.finalize()
        # return as an iterable, any number of pipelines may be defined!
        return [pm]


@numba.njit(pipeline_class=MyCompiler)
def test_unrolling(iterations):
    xs = list(range(1, iterations + 1))
    s = 0
    i = 0

    while i < len(xs):
        s += xs[i]
        i += 1

    return s


@numba.jit
def test_control(iterations):
    xs = list(range(1, iterations + 1))
    s = 0
    i = 0

    while i < len(xs):
        s += xs[i]
        i += 1

    return s


# if __name__ == "__main__":
#     # Warm up
#     for _ in range(3):
#         test_unrolling(10_000)
#         test_control(10_000)

#     rolled_times = []
#     unrolled_times = []

#     MAX_ITERATIONS = 500_000
#     TRIES = 100

#     iters_list = list(range(0, MAX_ITERATIONS + 1, 10_000))

#     for iters in iters_list:
#         rolled_total_time = 0
#         unrolled_total_time = 0

#         print(f"Benchmarking for {iters} iterations...")

#         # Test correctness
#         control_result = test_control(iters)
#         unrolling_result = test_unrolling(iters)
#         assert control_result == unrolling_result, f"Unrolling was incorrect for iters = {iters}"

#         # Benchmark
#         for _ in range(TRIES):
#             start = time.perf_counter()
#             test_unrolling(iters)
#             end = time.perf_counter()
#             unrolled_total_time += end - start

#             start = time.perf_counter()
#             test_control(iters)
#             end = time.perf_counter()
#             rolled_total_time += end - start

#         rolled_times.append(rolled_total_time / TRIES) 
#         unrolled_times.append(unrolled_total_time / TRIES) 
    
#     # Plot results
#     plt.plot(iters_list, rolled_times, label='Rolled Loop')
#     plt.plot(iters_list, unrolled_times, label='Unrolled Loop')
#     plt.xlabel("Iteration Count")
#     plt.ylabel("Avg Execution Time (seconds)")
#     plt.title("Loop Unrolling Benchmark")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
