import typing

import sympy
from sympy import Function

def generate_smt_file(loop_nodes, dep_cons, known_funcs, func_props):
    f_content = []
    f_content.append("(set-logic AUFNIRA)\n")
    generate_function_prop(func_props, f_content)
    generate_loop_bound_s_expressions_v2(loop_nodes, f_content)
    # print(dep_cons)
    generate_dep_constraints(dep_cons, f_content, known_funcs)
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
    for iv, bound in loop_nodes.items():
        iter_var = iv
        iter_var = str(iter_var).replace(".", "")
        f_content.append(f"(declare-const {iter_var} Int)\n")
        consts, funcs, sympy_expr = bound.get_sympy_expr(iter_var)
        for c in consts:
            if c not in defined_consts:
                defined_consts.add(c)
                f_content.append(f"(declare-const {c} Int)\n")
        known_funcs = {Function(f): typing.Callable[[int],int] for f in funcs if f is not None}
        smt_expr = sympy.smtlib_code(sympy_expr, symbol_table=known_funcs, auto_declare=False).split("\n")[0]
        f_content.append(smt_expr + "\n")

def generate_dep_constraints(dep_cons, f_content, known_funcs):
    symbol_table = {Function(f): typing.Callable[[int],int] for f in known_funcs}
    smt_expr = sympy.smtlib_code(dep_cons, auto_declare=False, symbol_table=symbol_table).split("\n")[0]
    f_content.append(smt_expr + "\n")


def generate_function_prop(f_props, f_content):
    declared_funcs = set()
    for func, props in f_props.items():
        if isinstance(func, str) and func not in declared_funcs:
            f_content.append(f"(declare-fun {func} (Int) Int)\n")
            declared_funcs.add(func)
        elif isinstance(func, tuple):
            if func[0] not in declared_funcs:
                f_content.append(f"(declare-fun {func[0]} (Int) Int)\n")
                declared_funcs.add(func[0])
            if func[1] not in declared_funcs:
                f_content.append(f"(declare-fun {func[1]} (Int) Int)\n")
                declared_funcs.add(func[1])
        for p in props:
            if p == 'monotonically_increasing':
                generate_function_prop_monotonicity(func, f_content)
            if p == 'triangularity' and isinstance(func, tuple) and len(func) == 2:
                generate_function_prop_triangularity(func[0], func[1], f_content)

def generate_function_prop_monotonicity(func_name, f_content):
    f_content.append(f"(assert ( forall ((x1 Int) (x2 Int)) (=> (> x2 x1) (>= ({func_name} x2) ({func_name} x1))) ))\n")

def generate_function_prop_triangularity(cmprs_func, idx_func, f_content):
    f_content.append(f"(assert ( forall ((x1 Int) (x2 Int)) (=> (> x2 ({cmprs_func} x1)) (> ({idx_func} x2) x1)) ))\n")

#TODO implement all the handy function properties



