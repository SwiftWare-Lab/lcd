
import numba
from numba.core.ir_utils import *
from numba.core import ir_utils


def block_reachability_analysis(cfg, start_block_label):
    # Perform reachability analysis to find all blocks reachable from start_block_label
    reachable = set()
    stack = [start_block_label]
    while stack:
        block_label = stack.pop()
        if block_label in reachable:
            continue
        reachable.add(block_label)
        for succ_label in cfg.successors(block_label):
            stack.append(succ_label)
    return reachable


def get_block_instructions(block):
    return [inst for inst in block.body if isinstance(inst, ir.Assign)]


def visualize_data_flow(func_ir):
    cfg = ir_utils.compute_cfg_from_blocks(func_ir.blocks)
    use_defs = compute_use_defs(func_ir.blocks)

    # Create a graph representing instructions as nodes and def-use relations as edges
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()

    # Add nodes for each instruction
    inst_list = []
    for block in func_ir.blocks.values():
        for inst in block.body:
            G.add_node(inst)
            inst_list.append(inst)

    # Add edges for def-use relations
    for block_label in cfg.topo_order():
        for def_var, use_vars in use_defs.usemap[block_label].items():
            for use_var in use_vars:
                G.add_edge(def_var, use_var)


    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue",
            font_size=8, font_weight="bold")
    plt.show()


def merge_sets(cur, coming):

    if cur is None:
        return coming
    if coming is None:
        return cur
    # copy current state to ret_state
    ret_state = cur.copy()
    # merge coming state to ret_state
    kk = coming.keys()
    for k in kk:
        if k not in ret_state.keys():
            ret_state[k] = coming[k]
    return ret_state  # Or your top value


def transfer_function(instr, in_state):
    """Example transfer function. Replace with your specific transfer function."""
    if isinstance(instr, ir.Assign):
        target = instr.target.name
        if isinstance(instr.value, ir.Const):
            out_state = in_state.copy()
            out_state[target] = instr.value.value
            return out_state
        else:
            out_state = in_state.copy()
            out_state[target] = "UU"
            return out_state
    return in_state.copy()


def compare_dicts(d1, d2):
    """Compares two dictionaries."""
    if len(d1) != len(d2):
        return False
    for k, v in d1.items():
        if k not in d2:
            return False
        if v != d2[k]:
            return False
    return True


def data_flow_analysis(func_ir):
    """Performs data flow analysis on a Numba function. This function sees
    all variables as strings.
    Args:
        func_ir: FunctionIR
    """
    # cres = compile_isolated(func, argtypes)
    # func_ir = cres.library.get_function_ir(func.name)
    cfg = ir_utils.compute_cfg_from_blocks(func_ir.blocks)

    block_data = {}
    initial_state = {"a": None, "b": None}

    # Initialize block data
    for block_label, block in func_ir.blocks.items():
        block_data[block_label] = {"in": initial_state.copy(),
                                   "out": initial_state.copy()}

    # Iterative worklist algorithm
    worklist = list(func_ir.blocks.keys())

    while worklist:
        block_label = worklist.pop(0)
        block = func_ir.blocks[block_label]

        old_in_state = block_data[block_label]["in"].copy()

        # Compute new in state
        new_in_state = initial_state.copy()
        for pred_label in cfg.predecessors(block_label):
            pred_out_state = block_data[pred_label[0]]["out"]
            new_in_state = merge_sets(new_in_state, pred_out_state)

        block_data[block_label]["in"] = new_in_state

        # Compute new out state
        current_state = new_in_state.copy()
        for instr in block.body:
            current_state = transfer_function(instr, current_state)
        block_data[block_label]["out"] = current_state

        # Check if out state changed
        #if block_data[block_label]["out"] != old_in_state:
        if compare_dicts(block_data[block_label]["out"], old_in_state):
            for succ_label in cfg.successors(block_label):
                if succ_label[0] not in worklist:
                    worklist.append(succ_label)

    return block_data