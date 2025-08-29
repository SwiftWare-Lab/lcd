import typing

import sympy
from sympy import Function

def generate_smt_file(loop_nodes, dep_cons, func_props):
    f_content = []
    f_content.append("(set-logic AUFNIRA)\n")
    generate_function_prop(func_props, f_content)
    generate_loop_bound_s_expressions_v2(loop_nodes, f_content)
    generate_dep_constraints(dep_cons, f_content)
    # generate_func_props(func_props, f_content)
    with open("smt_file.smt2", "w") as f:
        f.writelines(f_content)
    return '\n'.join(f_content)

def generate_loop_bound_s_expressions(iter_vars, bounds, f_content):
    defined_consts = set()
    headers = sorted(iter_vars.keys())
    for header in headers:
        iter_var = iter_vars[header]
        iter_var = str(iter_var).replace(".", "")
        f_content.append(f"(declare-const {iter_var} Int)\n")
        consts, funcs, sympy_expr = bounds[header].get_sympy_expr(iter_var)
        for c in consts:
            if c not in defined_consts:
                defined_consts.add(c)
                f_content.append(f"(declare-const {c} Int)\n")
        known_funcs = {Function(f): typing.Callable[[int],int] for f in funcs if f is not None}
        smt_expr = sympy.smtlib_code(sympy_expr, symbol_table=known_funcs, auto_declare=False).split("\n")[0]
        f_content.append(smt_expr + "\n")

def generate_loop_bound_s_expressions_v2(loop_nodes, f_content):
    defined_consts = set()
    # headers = sorted(iter_vars.keys())
    for loop_node in loop_nodes:
        iter_var = loop_node.iter_var
        iter_var = str(iter_var).replace(".", "")
        f_content.append(f"(declare-const {iter_var} Int)\n")
        consts, funcs, sympy_expr = loop_node.bound.get_sympy_expr(iter_var)
        for c in consts:
            if c not in defined_consts:
                defined_consts.add(c)
                f_content.append(f"(declare-const {c} Int)\n")
        known_funcs = {Function(f): typing.Callable[[int],int] for f in funcs if f is not None}
        smt_expr = sympy.smtlib_code(sympy_expr, symbol_table=known_funcs, auto_declare=False).split("\n")[0]
        f_content.append(smt_expr + "\n")

def generate_dep_constraints(dep_cons, f_content):
    smt_expr = sympy.smtlib_code(dep_cons, auto_declare=False).split("\n")[0]
    f_content.append(smt_expr + "\n")


def generate_function_prop(f_props, f_content):
    for func, props in f_props.items():
        f_content.append(f"(declare-fun {func} (Int) Int)\n")
        for p in props:
            if p == 'monotonically_increasing':
                generate_function_prop_monotonicity(func, f_content)

def generate_function_prop_monotonicity(func_name, f_content):
    f_content.append(f"(assert ( forall ((x1 Int) (x2 Int)) (=> (> x2 x1) (>= ({func_name} x2) ({func_name} x1))) ))\n")

#TODO implement all the handy function properties



