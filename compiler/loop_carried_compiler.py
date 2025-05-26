from pprint import pprint
from typing import Dict

from numba.core import ir_utils, config, ir
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
        func_ir: ir.FunctionIR = state.func_ir
        blocks: Dict[int, ir.Block] = func_ir.blocks
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

        # pprint(use_defs.usemap)
        # print()
        # pprint(use_defs.defmap)
        # print()
        # pprint(live_map)
        # print()
        # pprint()

        # todo: get intersection of R and W
        #   todo: gonna do this by iterating over all pairs of R intersect W and the one that happens later in the pair is the "after"
        # todo: only go through each label once

        reads, writes = {}, {} # variable: (label, statement_index)
        temp2val = {} # maps numba generated temporary variables to its value 
        for loop in cfg.loops().values():
            for label in loop.body:
                loop_block = blocks[label]
                for i, st in enumerate(loop_block.body):
                    if isinstance(st, ir.Assign) and isinstance(st.value, ir.Expr):
                        match st.value.op:
                            case 'getitem':
                                # * Assumes getitem always used to load a value from a list to a temporary variable
                                temp2val[st.target] = st.value.value
                            case 'binop' | 'inplace_binop':
                                if st.value.lhs in temp2val:
                                    val = temp2val[st.value.lhs]
                                    if val in reads:
                                        reads[val].add((label, i))
                                    else:
                                        reads[val] = {(label, i)}
                                if st.value.rhs in temp2val:
                                    val = temp2val[st.value.rhs]
                                    if val in reads:
                                        reads[val].add((label, i))
                                    else:
                                        reads[val] = {(label, i)}
                    elif isinstance(st, ir.SetItem):
                        if st.target in writes:
                            writes[st.target].add((label, i))
                        else:
                            writes[st.target] = {(label, i)}
        print('=======')
        print("Reads:")
        pprint(reads)
        print('=======')
        print("Writes:")
        pprint(writes)

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

