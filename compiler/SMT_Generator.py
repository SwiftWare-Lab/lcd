import re
import typing

import sympy
from sympy import Function
from sympy.printing.smtlib import SMTLibPrinter

from compiler.spf_def import (
    AndPremise,
    DependencePremise,
    ExpressionPremise,
    ForAllPremise,
    NotPremise,
    OrPremise,
    Premise,
    SubstitutePremise,
)

def generate_smt_file(loop_nodes, dep_cons, known_funcs, func_props):
    f_content = []
    f_content.append("(set-logic AUFNIRA)\n")
    # Function declarations must precede loop-bound assertions that call the
    # functions.  Property assertions come afterwards because relational
    # properties may refer to symbolic loop extents such as ``n``.
    generate_function_declarations(func_props, f_content)
    generate_loop_bound_s_expressions_v2(loop_nodes, f_content)
    generate_function_prop_assertions(func_props, f_content)
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
    if isinstance(dep_cons, Premise):
        existing_declarations = _declared_names(f_content)
        for declaration in premise_smt_declarations(
                dep_cons,
                known_funcs,
        ):
            declared_name = _declaration_name(declaration)
            if declared_name not in existing_declarations:
                f_content.append(declaration + "\n")
                existing_declarations.add(declared_name)
        f_content.append(
            f"(assert {premise_to_smt(dep_cons, known_funcs)})\n"
        )
        return
    symbol_table = {Function(f): typing.Callable[[int],int] for f in known_funcs}
    smt_expr = sympy.smtlib_code(dep_cons, auto_declare=False, symbol_table=symbol_table).split("\n")[0]
    f_content.append(smt_expr + "\n")


def premise_to_smt(premise, known_funcs=()):
    """Emit an SPF premise as an SMT-LIB Boolean expression.

    Quantifiers are emitted directly because released SymPy versions do not
    provide a first-order ``ForAll`` expression. Atomic expressions continue
    to use SymPy's SMT printer so their established arithmetic and access
    lowering is preserved.
    """
    if isinstance(premise, AndPremise):
        return _nary_smt(
            "and",
            [premise_to_smt(child, known_funcs) for child in premise.premises],
            empty="true",
        )
    if isinstance(premise, OrPremise):
        return _nary_smt(
            "or",
            [premise_to_smt(child, known_funcs) for child in premise.premises],
            empty="false",
        )
    if isinstance(premise, NotPremise):
        return f"(not {premise_to_smt(premise.premise, known_funcs)})"
    if isinstance(premise, ForAllPremise):
        bounds = _nary_smt(
            "and",
            [premise_to_smt(bound, known_funcs) for bound in premise.bounds],
            empty="true",
        )
        body = premise_to_smt(premise.premise, known_funcs)
        return (
            f"(forall (({premise.quantified_var} Int)) "
            f"(=> {bounds} {body}))"
        )
    if isinstance(
            premise,
            (DependencePremise, ExpressionPremise, SubstitutePremise),
    ):
        expression, premise_funcs = premise.get_sympy_expression()
        all_funcs = set(known_funcs)
        all_funcs.update(premise_funcs)
        symbol_table = {
            Function(function): typing.Callable[[int], int]
            for function in all_funcs
        }
        return SMTLibPrinter(symbol_table=symbol_table).doprint(expression)
    raise TypeError(
        f"Unsupported SPF premise for SMT emission: {type(premise).__name__}."
    )


def premise_smt_declarations(premise, known_funcs=()):
    """Return declarations needed by a directly emitted premise."""
    expressions = []
    premise_funcs = set(known_funcs)
    quantified_names = set()
    _collect_atomic_expressions(
        premise,
        expressions,
        premise_funcs,
        quantified_names,
    )
    if not expressions:
        return []
    expression = sympy.And(*expressions)
    symbol_table = {
        Function(function): typing.Callable[[int], int]
        for function in premise_funcs
    }
    declaration_source = sympy.smtlib_code(
        expression,
        auto_declare=True,
        symbol_table=symbol_table,
    )
    declarations = []
    for line in declaration_source.splitlines():
        if not line.startswith("(declare-"):
            continue
        if _declaration_name(line) in quantified_names:
            continue
        declarations.append(line)
    return declarations


def _collect_atomic_expressions(
        premise,
        expressions,
        known_funcs,
        quantified_names,
):
    if isinstance(premise, (AndPremise, OrPremise)):
        for child in premise.premises:
            _collect_atomic_expressions(
                child,
                expressions,
                known_funcs,
                quantified_names,
            )
        return
    if isinstance(premise, NotPremise):
        _collect_atomic_expressions(
            premise.premise,
            expressions,
            known_funcs,
            quantified_names,
        )
        return
    if isinstance(premise, ForAllPremise):
        quantified_names.add(premise.quantified_var)
        for bound in premise.bounds:
            _collect_atomic_expressions(
                bound,
                expressions,
                known_funcs,
                quantified_names,
            )
        _collect_atomic_expressions(
            premise.premise,
            expressions,
            known_funcs,
            quantified_names,
        )
        return
    if isinstance(
            premise,
            (DependencePremise, ExpressionPremise, SubstitutePremise),
    ):
        expression, expression_funcs = premise.get_sympy_expression()
        expressions.append(expression)
        known_funcs.update(expression_funcs)
        return
    raise TypeError(
        f"Unsupported SPF premise for SMT declarations: "
        f"{type(premise).__name__}."
    )


def _declared_names(f_content):
    names = set()
    for line in f_content:
        if line.lstrip().startswith("(declare-"):
            names.add(_declaration_name(line.lstrip()))
    return names


def _declaration_name(declaration):
    match = re.match(r"\(declare-(?:const|fun)\s+([^\s()]+)", declaration)
    if match is None:
        raise ValueError(f"Malformed SMT declaration: {declaration!r}.")
    return match.group(1)


def _nary_smt(operator, operands, empty):
    if not operands:
        return empty
    if len(operands) == 1:
        return operands[0]
    return f"({operator} {' '.join(operands)})"


def generate_function_declarations(f_props, f_content):
    declared_funcs = set()
    for func in f_props:
        function_names = (func,) if isinstance(func, str) else func
        for function_name in function_names:
            if function_name not in declared_funcs:
                f_content.append(
                    f"(declare-fun {function_name} (Int) Int)\n"
                )
                declared_funcs.add(function_name)


def generate_function_prop_assertions(f_props, f_content):
    for func, props in f_props.items():
        for p in props:
            property_name, property_args = _property_parts(p)
            if property_name == 'monotonically_increasing':
                if not isinstance(func, str):
                    raise ValueError(
                        "monotonically_increasing expects one function"
                    )
                generate_function_prop_monotonicity(func, f_content)
            elif property_name == 'triangularity':
                if not isinstance(func, tuple) or len(func) != 2:
                    raise ValueError(
                        "triangularity expects a pair of functions"
                    )
                generate_function_prop_triangularity(func[0], func[1], f_content)
            elif property_name == 'segmented_monotonically_increasing':
                if not isinstance(func, tuple) or len(func) != 2:
                    raise ValueError(
                        "segmented_monotonically_increasing expects "
                        "(pointer_function, index_function)"
                    )
                if len(property_args) != 1:
                    raise ValueError(
                        "segmented_monotonically_increasing expects the "
                        "number of segments"
                    )
                generate_function_prop_segmented_monotonicity(
                    func[0],
                    func[1],
                    property_args[0],
                    f_content,
                )


def generate_function_prop(f_props, f_content):
    """Emit declarations and assertions for callers of the legacy helper."""
    generate_function_declarations(f_props, f_content)
    generate_function_prop_assertions(f_props, f_content)


def _property_parts(property_spec):
    if isinstance(property_spec, str):
        return property_spec, ()
    if isinstance(property_spec, tuple) and property_spec:
        return property_spec[0], property_spec[1:]
    raise ValueError(f"Invalid function property: {property_spec!r}")

def generate_function_prop_monotonicity(func_name, f_content):
    f_content.append(f"(assert ( forall ((x1 Int) (x2 Int)) (=> (> x2 x1) (>= ({func_name} x2) ({func_name} x1))) ))\n")

def generate_function_prop_triangularity(cmprs_func, idx_func, f_content):
    f_content.append(f"(assert ( forall ((x1 Int) (x2 Int)) (=> (> x2 ({cmprs_func} x1)) (> ({idx_func} x2) x1)) ))\n")


def generate_function_prop_segmented_monotonicity(
        pointer_func,
        index_func,
        segment_count,
        f_content,
):
    """Assert that indices are nondecreasing inside each sparse segment."""
    if segment_count not in _declared_names(f_content):
        f_content.append(f"(declare-const {segment_count} Int)\n")
    f_content.append(
        "(assert (forall ((segment Int) (p Int) (q Int)) "
        "(=> (and "
        f"(<= 0 segment) (< segment {segment_count}) "
        f"(<= ({pointer_func} segment) p) (< p q) "
        f"(< q ({pointer_func} (+ segment 1)))) "
        f"(<= ({index_func} p) ({index_func} q)))))\n"
    )

#TODO implement all the handy function properties
