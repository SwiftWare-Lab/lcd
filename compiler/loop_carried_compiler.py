from numba.typed import Dict

from numba import njit, prange
from numba.core import ir, ir_utils, config, errors
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import *
from numba.core.untyped_passes import ReconstructSSA, FixupArgs, TranslateByteCode
from numba.extending import type_callable
import numpy as np
import numba
from numba.core.analysis import compute_live_map, compute_dead_maps, compute_live_variables
from compiler.analysis import *
from graphviz import Source

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
        data_values, data_blocks = extract_data_value(blocks)

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

        # This is like use_defs from numba but with symbolised values
        # flabels = extract_fused_loops(blocks)

        # print("Fused Loop Labels: ", flabels)
        loop_bodies, lp_pars = find_all_loop_bodies(cfg)
        print("Loop Bodies: ", loop_bodies)
        iter_vars = extract_iter_variables(blocks, loop_bodies, lp_pars)
        print("Iterators: ", iter_vars)
        floops = find_to_be_fused_loops(data_values, iter_vars)
        print("Fusable Loops: ", floops)
        loop_header_bodies = extract_loop_header_body_map(loop_bodies, lp_pars)
        print("Loop Header Bodies: ", loop_header_bodies)
        spf_tree = create_spf_tree(lp_pars, loop_header_bodies)
        print(spf_tree)
        accesses, stmnt_labels = extract_read_write_accesses(blocks, data_values, loop_header_bodies, iter_vars)
        print("Statement Labels: ", stmnt_labels)
        print("Symbolic Accesses: ", accesses)
        latest_accesses = {}
        for l in floops:
            latest_accesses.update(extract_latest_write_accesses_in_loop(l, accesses['writes'], loop_bodies))
            print("LOOP", l, ":", latest_accesses)
        schedule = create_statement_schedule(stmnt_labels, spf_tree, iter_vars)
        print(schedule)
        raw_dependencies = extract_raw_data_dependences(accesses['reads'], latest_accesses, schedule, loop_header_bodies)
        fused_iter_vars = [x.name for key, x in iter_vars.items() if key in floops]
        dependencies, ex_cons = refine_dependences(raw_dependencies, fused_iter_vars)

        for i, dependence in enumerate(dependencies):
            if dependence.type == 'RAW':
                print(dependence)
                print(ex_cons[i])
                print("-----------------")


        # dot.render("cfg_diagram", view=True)

        order = find_topo_order(blocks)
        return mutate  # the pass has not mutated the IR



class MyCompiler(CompilerBase):  # custom compiler extends from CompilerBase

    def define_pipelines(self):
        # define a new set of pipelines (just one in this case) and for ease
        # base it on an existing pipeline from the DefaultPassBuilder,
        # namely the "nopython" pipeline
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        # Add the new pass to run after IRProcessing
        pm.add_pass_after(LoopFusion1, ReconstructSSA)
        #pm.add_pass_after(ParFdd, IRProcessing)
        # finalize
        pm.finalize()
        # return as an iterable, any number of pipelines may be defined!
        return [pm]

