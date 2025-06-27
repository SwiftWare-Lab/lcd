from typing import List
import copy

from numba.core import ir, ir_utils


def extract_data_value(blocks):
    data_values = {}
    data_block = {}
    for label, block in blocks.items():
        for st in block.body:
            if isinstance(st, ir.Assign):
                data_values[st.target.name] = st.value
                data_block[st.target.name] = label
    return data_values, data_block


def get_increment_and_final(loops, blocks, data_values):
    """Parses loop blocks to find the loop increment and final value"""
    for loop in loops:
        header_block = blocks[loop.header]
        print(type(header_block.body[0].value.value))
    
    return 0, 0


def unroll_loop(loops, blocks, data_values, factor):
    """Performs loop unrolling, modifying the IR through blocks. Assumes loops consist of only 1 basic block.
    In while loops, assumes the loop variable increment is the last statement.
    """
    changed = False

    for loop in loops:
        unrolled = []
        if is_for_loop(blocks[loop.header].body):
            # Get increment statements from header
            header = blocks[loop.header].body
            iternext_stmt = header[0]
            get_next_val_stmt = header[1]
            iternext_assign = header[3]
            incr_stmts = [iternext_stmt, get_next_val_stmt, iternext_assign]

            # Reader unroll_part factor - 1 times, then append the body as is
            body_label = [b for b in loop.body if b != loop.header][0] # body consists of header and body itsel
            body_stmts = blocks[body_label].body
            # Add first iteration as is (excluding jump) so that loop var is assigned same phi var from loop header
            unrolled += body_stmts[:-1]

            # Add increment statements and loop excluded jump, this part is repeated
            unroll_part = incr_stmts + body_stmts[:-1]
            for _ in range(factor - 1):
                unrolled += ssa_rename(unroll_part, varmap={})
            # Add jump
            unrolled.append(body_stmts[-1])
            changed |= unrolled != body_stmts
            # Replace rolled loop with unrolled
            blocks[body_label].body = unrolled
        else:
            # Find increment statement
            # What we expect the "body" of a while loop to be, Numba calls it the header of the while loop
            stmts = blocks[loop.header].body
            branch_statement = stmts[-1]
            cond_var_name = data_values[branch_statement.cond.name].args[0].name
            cond = data_values[cond_var_name]
            loop_var = cond.lhs
            final = cond.rhs

            incr_assign_idx = [i for i, st in enumerate(stmts) if isinstance(st, ir.Assign) and st.target == loop_var][-1]
            incr_expr = stmts[incr_assign_idx - 1].value
            incr_var = incr_expr.lhs if incr_expr.lhs != loop_var else incr_expr.rhs
            incr = data_values[incr_var.name].value

            unroll_part = stmts[: incr_assign_idx + 1] # +1 to include the increment
            # Unroll loop with renaming to maintain SSA form
            for _ in range(factor):
                unrolled += ssa_rename(unroll_part, varmap={})
            # Add branching logic
            unrolled += stmts[incr_assign_idx + 1 :]
            changed |= unrolled != stmts
            # Replace rolled loop with unrolled
            blocks[loop.header].body = unrolled

    return changed


def is_for_loop(header):
    """Determines if a loop is a for loop. If not, it is a while loop"""
    return len(header) == 5 and \
           isinstance(header[0], ir.Assign) and \
           isinstance(header[0].value, ir.Expr) and \
           header[0].value.op == 'iternext'
    

def ssa_rename(stmts, varmap):
    new_stmts = []
    for st in stmts:
        st = copy.deepcopy(st)
        if isinstance(st, ir.Assign):
            # Create new var name if if it is a numba variable (user-defined variables should not be rewritten)
            old_target = st.target.name
            if old_target.startswith('$'):
                new_target = ir_utils.mk_unique_var(old_target)
                # Add to map
                varmap[old_target] = new_target
                # Update statement
                st.target.name = new_target
            # Need to rewrite rhs either way so that numba variable usages are updated
            st.value = rewrite_expr(st.value, varmap)
        new_stmts.append(st)
    return new_stmts


def rewrite_expr(expr, varmap):
    expr = copy.deepcopy(expr)

    def rewrite_var(v):
        if isinstance(v, ir.Var) and v.name in varmap:
            return ir.Var(scope=v.scope, name=varmap[v.name], loc=v.loc)
        return v
    
    if isinstance(expr, ir.Var):
        return rewrite_var(expr)
    
    if isinstance(expr, ir.Expr):
        for attr in expr._kws.keys():
            val = getattr(expr, attr)
            if isinstance(val, ir.Var):
                setattr(expr, attr, rewrite_var(val))
            elif isinstance(val, list):
                setattr(expr, attr, [rewrite_var(v) for v in val])
        return expr
    
    return expr