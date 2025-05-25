from pprint import pprint

from numba.core import ir_utils, config
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.untyped_passes import IRProcessing
import numba.core.analysis as numba_analysis


config.DEBUG_ARRAY_OPT = 3


@register_pass(mutates_CFG=True, analysis_only=True)
class LoopFusion1(FunctionPass):
    _name = "loop_fusion1"

    def __init__(self):
        super().__init__(self)

    def run_pass(self, state):
        mutate = False
        func_ir = state.func_ir
        blocks = func_ir.blocks
        ir_utils.dprint_func_ir(func_ir, "before maximize fusion down")
        cfg = ir_utils.compute_cfg_from_blocks(blocks)
        use_defs = ir_utils.compute_use_defs(func_ir.blocks)

        # plot cfg
        graphiz_digraph = cfg.render_dot("diag_cfg.dot")
        # convert graphvizdigraph to pdf
        graphiz_digraph.format = "pdf"
        #graphiz_digraph.view()
        cfg.dump()

        live_map = numba_analysis.compute_live_map(cfg, func_ir.blocks, use_defs.usemap, use_defs.defmap)

        pprint(use_defs.usemap)
        print()
        pprint(use_defs.defmap)
        print()
        pprint(live_map)
        print()

        # * Goal: find intersections of reads and writes
        # todo 1: find all reads and writes
        # ? is this the approach: at a R/W, find if that same statement OR iteration is also stored as a W/R?

        return mutate  # the pass has not mutated the IR


class MyCompiler(CompilerBase):  # custom compiler extends from CompilerBase
    def define_pipelines(self):
        # define a new set of pipelines (just one in this case) and for ease
        # base it on an existing pipeline from the DefaultPassBuilder,
        # namely the "nopython" pipeline
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        # Add the new pass to run after IRProcessing
        pm.add_pass_after(LoopFusion1, IRProcessing)
        #pm.add_pass_after(ParFdd, IRProcessing)
        # finalize
        pm.finalize()
        # return as an iterable, any number of pipelines may be defined!
        return [pm]

