from numba.core import ir_utils, config
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import *
from numba.core.untyped_passes import IRProcessing
from numba.core.analysis import compute_live_map, compute_dead_maps, compute_live_variables

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
        cfg.dump()

        data_values, _ = extract_data_value(blocks)
        mutate = unroll_simple_loops(cfg, blocks, data_values, factor=4)
        # todo: currently assumes factor is a multiple of final, and that final >= factor
        # todo: get init for while loops, get init, incr and final for for loops, create new loop for remaining iterations. Must learn how to add blocks in numba

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
def test_loop():
    xs = [1,2,3,]
    s = 0
    i = 0
    # for j in range(len(xs)):
    #     s += xs[j]
    while i < len(xs):
        s += xs[i]
        i += 1
    # for _ in range(10):
    #     for i in range(len(xs)):
    #         s += xs[i]
    # for _ in range(10):
    #     for _ in range(10):
    #         for _ in range(10):
    #             for _ in range(10):
    #                 for i in range(len(xs)):
    #                     s += xs[i]
    return s


def test_loop_og():
    xs = [1,2,3,4,5,6,7,8,9,10]
    s = 0
    i = 0
    # for i in range(len(xs)):
    #     s += xs[i]
    while i < len(xs):
        s += xs[i]
        i += 1
    # for _ in range(10):
    #     for i in range(len(xs)):
    #         s += xs[i]
    # for _ in range(10):
    #     for _ in range(10):
    #         for _ in range(10):
    #             for _ in range(10):
    #                 for i in range(len(xs)):
    #                     s += xs[i]
    return s


if __name__ == "__main__":
    # print(test_loop_og())
    print(test_loop())
    pass