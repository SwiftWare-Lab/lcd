from typing import List
import copy
import operator

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


def find_defs(blocks, varname):
    defs = []
    for label, block in blocks.items():
        for st in block.body:
            if isinstance(st, ir.Assign):
                if st.target.name == varname:
                    defs.append((label, st))
    return defs


def copy_loc(loc: ir.Loc) -> ir.Loc:
    return ir.Loc(loc.filename, loc.line)


def get_increment_and_final(loops, blocks, data_values):
    """Parses loop blocks to find the loop increment and final value"""
    for loop in loops:
        header_block = blocks[loop.header]
        print(type(header_block.body[0].value.value))
    
    return 0, 0


def unroll_simple_loops(cfg, blocks, data_values, factor):
    """Performs loop unrolling, modifying the IR through blocks. Assumes loops consist of only 1 basic block.
    In while loops, assumes the loop variable increment is the last statement.
    """
    # body size 2 because numba includes header in body
    simple_loops = [loop for loop in cfg.loops().values() if len(loop.body) == 2]

    changed = False

    # Todo: 
    # - get init, final, and incr in for loops
    # - no need for init value in while loops, just use the loop var itself before loop starts 
    # - adjust unrolled loop to iterate  up to max multiple of factor
    # - add rolled loop after for remaining iterations

    for loop in simple_loops:
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
            # Get needed blocks
            # What we expect the "body" of a while loop to be, Numba calls it the header of the while loop
            header_block = copy.deepcopy(blocks[loop.header])
            # Condition filters out back edges, and index 0 because we assume only 1 block leads to the loop
            pred_label = [label for label in cfg._preds[loop.header] if label < loop.header][0]
            pred_block = copy.deepcopy(blocks[pred_label])
            stmts = header_block.body

            # Find loop variable, final iteration, and increment statement
            cond = get_loop_cond(header_block, data_values)
            pred_cond = get_loop_cond(pred_block, data_values)
            loop_var = pred_cond.lhs
            final = pred_cond.rhs
            incr_assign_idx = [i for i, st in enumerate(stmts) if isinstance(st, ir.Assign) and st.target == loop_var][-1]
            incr_expr = stmts[incr_assign_idx - 1].value
            incr_var = incr_expr.lhs if incr_expr.lhs != loop_var else incr_expr.rhs
            incr = data_values[incr_var.name]

            # Change condition so iterate unrolled loop does not go out of bounds if iterations is not a multiple of factor and increment
            stmt_before_cond = header_block.body[-5]
            pred_stmt_before_cond = pred_block.body[-5]
            # At this point of code, loop hasn't begun thus init value is stored in loop var
            # Create statement to calculate
            loc = copy_loc(pred_stmt_before_cond.loc)
            scope = pred_stmt_before_cond.target.scope

            # Copy increment variable to the pred block
            # Todo: remove it from loop body
            pred_incr_assign = ir.Assign(
                incr,
                ir.Var(scope, ir_utils.mk_unique_var("$incr"), loc),
                loc)
            pred_block.insert_after(pred_incr_assign, pred_stmt_before_cond)
            # Store factor in a var
            factor_assign = ir.Assign(
                ir.Const(factor, loc),
                ir.Var(scope, ir_utils.mk_unique_var("$factor"), loc),
                loc)
            pred_block.insert_after(factor_assign, pred_incr_assign)
            # Calculate how much loop var increments after an iteration
            incr_per_iter = ir.Assign(
                ir.Expr.binop(operator.mul, factor_assign.target, pred_incr_assign.target, loc),
                ir.Var(scope, ir_utils.mk_unique_var("$incr_per_iter"), loc),
                loc)
            pred_block.insert_after(incr_per_iter, factor_assign)
            # Subtract 1 since increment happens before checking condition. Otherwise, if total iters is a multiple of factor and incr, last iteration would be missed
            const1 = ir.Assign(
                ir.Const(1, loc),
                ir.Var(scope, ir_utils.mk_unique_var("$const1"), loc),
                loc)
            pred_block.insert_after(const1, incr_per_iter)
            max_step_offset = ir.Assign(
                ir.Expr.binop(operator.sub, incr_per_iter.target, const1.target, loc),
                ir.Var(scope, ir_utils.mk_unique_var("$max_step_offset"), loc),
                loc)
            pred_block.insert_after(max_step_offset, const1)
            pred_new_loop_cond_lhs = ir.Assign(
                ir.Expr.binop(operator.add, loop_var, max_step_offset.target, loc),
                ir.Var(scope, ir_utils.mk_unique_var("$new_pred_cond_lhs"), loc),
                loc)
            pred_block.insert_after(pred_new_loop_cond_lhs, max_step_offset)
            # This goes into the loop as well to recalculate each iteration
            new_loop_cond_lhs = ir.Assign(
                ir.Expr.binop(operator.add, loop_var, max_step_offset.target, loc),
                ir.Var(scope, ir_utils.mk_unique_var("$new_cond_lhs"), loc),
                loc)
            header_block.insert_after(new_loop_cond_lhs, stmt_before_cond)

            pred_cond.lhs = pred_new_loop_cond_lhs.target
            cond.lhs = new_loop_cond_lhs.target

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
    

def get_loop_cond(block, data_values):
    branch_stmt = block.body[-1]
    cond_var_name = data_values[branch_stmt.cond.name].args[0].name
    return data_values[cond_var_name]
    

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
