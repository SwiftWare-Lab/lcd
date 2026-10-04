import ast
from copy import deepcopy

import sympy


def _is_access(value):
    from compiler.spf_raising import Access

    return isinstance(value, Access)


def _is_expression(value):
    from compiler.spf_raising import Expression

    return isinstance(value, Expression)


def _subscript_value(value, suffix, iteration_vars=None):
    if _is_access(value):
        return value.subscript_index(suffix, iteration_vars)
    if _is_expression(value):
        return value.subscript(suffix, iteration_vars)
    if (isinstance(value, str) and value.isidentifier()
            and (iteration_vars is None or value in iteration_vars)):
        return value + suffix
    return value


def _sympy_value(value):
    known_funcs = set()
    if _is_access(value):
        funcs, expr = value.get_sympy_expr()
        known_funcs.update(funcs)
        return expr, known_funcs
    if _is_expression(value):
        return value.get_sympy_expr(), known_funcs
    if isinstance(value, str):
        if value.isnumeric():
            return int(value), known_funcs
        try:
            literal = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            literal = None
        if isinstance(literal, (int, float)) and not isinstance(literal, bool):
            return sympy.sympify(literal), known_funcs
        return sympy.Symbol(value.replace(".", "")), known_funcs
    return value, known_funcs


def _collect_read_accesses(value, reads):
    if _is_access(value):
        reads.append(value)
        return
    if _is_expression(value):
        for element in value.elements:
            _collect_read_accesses(element, reads)


def _replace_value(value, replacements):
    if _is_access(value):
        return type(value)(_replace_value(value.var, replacements), _replace_value(value.index, replacements))
    if _is_expression(value):
        return type(value)([_replace_value(element, replacements) for element in value.elements])
    if isinstance(value, str) and value in replacements:
        return replacements[value]
    return value


class Premise:
    def subscript(self, suffix=None, iteration_vars=None, right_iteration_vars=None):
        raise NotImplementedError("Subclasses must implement subscript.")

    def get_sympy_expression(self):
        raise NotImplementedError("Subclasses must implement get_sympy_expression.")

    def read_accesses(self):
        return []

    def copy(self):
        return deepcopy(self)


class DependencePremise(Premise):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def subscript(self, suffix=None, iteration_vars=None, right_iteration_vars=None):
        left_suffix = suffix or "1"
        right_suffix = suffix or "2"
        if right_iteration_vars is None:
            right_iteration_vars = iteration_vars
        self.left = _subscript_value(self.left, left_suffix, iteration_vars)
        self.right = _subscript_value(self.right, right_suffix, right_iteration_vars)
        return self

    def __eq__(self, other):
        if type(other) == type(self):
            return other.left == self.left and other.right == self.right
        else:
            return False

    def get_sympy_expression(self):
        raise NotImplementedError("This functionality is only implemented for subclass objects.")

    def get_sympy_expressions(self, func):
        left_sym, known_funcs = _sympy_value(self.left)
        right_sym, right_funcs = _sympy_value(self.right)
        known_funcs.update(right_funcs)
        return func(left_sym, right_sym), known_funcs

    def read_accesses(self):
        reads = []
        _collect_read_accesses(self.left, reads)
        _collect_read_accesses(self.right, reads)
        return reads

    def __repr__(self):
        raise NotImplementedError("This functionality is only implemented for subclass objects.")

    def __str__(self):
        return self.__repr__()

class Equality(DependencePremise):
    def get_sympy_expression(self):
        return self.get_sympy_expressions(sympy.Eq)

    def __repr__(self):
        return str(self.left) + ' = ' + str(self.right)

    def __eq__(self, other):
        if type(other) == type(self):
            return (other.left == self.left and other.right == self.right) or \
                (other.left == self.right and other.right == self.left)
        else:
            return False

class GreaterThanOrEqual(DependencePremise):
    def get_sympy_expression(self):
        return self.get_sympy_expressions(sympy.GreaterThan)

    def __repr__(self):
        return str(self.left) + ' >= ' + str(self.right)


class GreaterThan(DependencePremise):
    def get_sympy_expression(self):
        return self.get_sympy_expressions(sympy.Gt)

    def __repr__(self):
        return str(self.left) + ' > ' + str(self.right)


class LessThanOrEqual(DependencePremise):
    def get_sympy_expression(self):
        return self.get_sympy_expressions(sympy.LessThan)

    def __repr__(self):
        return str(self.left) + ' <= ' + str(self.right)


class LessThan(DependencePremise):
    def get_sympy_expression(self):
        return self.get_sympy_expressions(sympy.Lt)

    def __repr__(self):
        return str(self.left) + ' < ' + str(self.right)


class NotEqual(DependencePremise):
    def get_sympy_expression(self):
        return self.get_sympy_expressions(sympy.Ne)

    def __repr__(self):
        return str(self.left) + ' != ' + str(self.right)


class ExpressionPremise(Premise):
    def __init__(self, condition):
        self.condition = condition

    def subscript(self, suffix=None, iteration_vars=None, right_iteration_vars=None):
        if suffix is not None:
            self.condition = _subscript_value(self.condition, suffix, iteration_vars)
        return self

    def get_sympy_expression(self):
        return _sympy_value(self.condition)

    def read_accesses(self):
        reads = []
        _collect_read_accesses(self.condition, reads)
        return reads

    def __repr__(self):
        return str(self.condition)

    def __str__(self):
        return self.__repr__()


class AndPremise(Premise):
    def __init__(self, premises):
        self.premises = premises

    def subscript(self, suffix=None, iteration_vars=None, right_iteration_vars=None):
        for premise in self.premises:
            premise.subscript(suffix, iteration_vars, right_iteration_vars)
        return self

    def get_sympy_expression(self):
        condition = sympy.true
        known_funcs = set()
        for premise in self.premises:
            premise_expr, premise_funcs = premise.get_sympy_expression()
            known_funcs.update(premise_funcs)
            condition = sympy.And(condition, premise_expr)
        return condition, known_funcs

    def read_accesses(self):
        reads = []
        for premise in self.premises:
            reads.extend(premise.read_accesses())
        return reads

    def __repr__(self):
        return " and ".join(map(str, self.premises))

    def __str__(self):
        return self.__repr__()


class OrPremise(Premise):
    def __init__(self, premises):
        self.premises = premises

    def subscript(self, suffix=None, iteration_vars=None, right_iteration_vars=None):
        for premise in self.premises:
            premise.subscript(suffix, iteration_vars, right_iteration_vars)
        return self

    def get_sympy_expression(self):
        condition = sympy.false
        known_funcs = set()
        for premise in self.premises:
            premise_expr, premise_funcs = premise.get_sympy_expression()
            known_funcs.update(premise_funcs)
            condition = sympy.Or(condition, premise_expr)
        return condition, known_funcs

    def read_accesses(self):
        reads = []
        for premise in self.premises:
            reads.extend(premise.read_accesses())
        return reads

    def __repr__(self):
        return " or ".join(map(str, self.premises))

    def __str__(self):
        return self.__repr__()


class NotPremise(Premise):
    def __init__(self, premise):
        self.premise = premise

    def subscript(self, suffix=None, iteration_vars=None, right_iteration_vars=None):
        self.premise.subscript(suffix, iteration_vars, right_iteration_vars)
        return self

    def get_sympy_expression(self):
        premise_expr, known_funcs = self.premise.get_sympy_expression()
        return sympy.Not(premise_expr), known_funcs

    def read_accesses(self):
        return self.premise.read_accesses()

    def __repr__(self):
        return "not (" + str(self.premise) + ")"

    def __str__(self):
        return self.__repr__()


class SubstitutePremise(Premise):
    def __init__(self, premise, replacements):
        self.premise = premise
        self.replacements = replacements

    def subscript(self, suffix=None, iteration_vars=None, right_iteration_vars=None):
        self.premise.subscript(suffix, iteration_vars, right_iteration_vars)
        if suffix is not None:
            self.replacements = {
                _subscript_value(source, suffix, iteration_vars):
                    _subscript_value(target, suffix, iteration_vars)
                for source, target in self.replacements.items()
            }
        return self

    def get_sympy_expression(self):
        premise_expr, known_funcs = self.premise.get_sympy_expression()
        sympy_replacements = {}
        for source, target in self.replacements.items():
            source_sym, source_funcs = _sympy_value(source)
            target_sym, target_funcs = _sympy_value(target)
            known_funcs.update(source_funcs)
            known_funcs.update(target_funcs)
            sympy_replacements[source_sym] = target_sym
        return premise_expr.subs(sympy_replacements), known_funcs

    def read_accesses(self):
        substituted = self.copy()
        substituted.premise = _replace_premise_values(substituted.premise, substituted.replacements)
        return substituted.premise.read_accesses()

    def __repr__(self):
        mapping = ", ".join(f"{source} -> {target}" for source, target in self.replacements.items())
        return str(self.premise) + " with " + mapping

    def __str__(self):
        return self.__repr__()


def _replace_premise_values(premise, replacements):
    if isinstance(premise, DependencePremise):
        premise.left = _replace_value(premise.left, replacements)
        premise.right = _replace_value(premise.right, replacements)
    elif isinstance(premise, ExpressionPremise):
        premise.condition = _replace_value(premise.condition, replacements)
    elif isinstance(premise, (AndPremise, OrPremise)):
        premise.premises = [_replace_premise_values(child, replacements) for child in premise.premises]
    elif isinstance(premise, NotPremise):
        premise.premise = _replace_premise_values(premise.premise, replacements)
    elif isinstance(premise, ForAllPremise):
        premise.bounds = [_replace_premise_values(bound, replacements) for bound in premise.bounds]
        premise.premise = _replace_premise_values(premise.premise, replacements)
    elif isinstance(premise, SubstitutePremise):
        premise.premise = _replace_premise_values(premise.premise, replacements)
    return premise


class ForAllPremise(Premise):
    def __init__(self, quantified_var, bounds, premise):
        self.quantified_var = quantified_var
        self.bounds = bounds
        self.premise = premise

    def subscript(self, suffix=None, iteration_vars=None, right_iteration_vars=None):
        if suffix is not None:
            scoped_iteration_vars = iteration_vars
            if iteration_vars is not None:
                scoped_iteration_vars = set(iteration_vars)
                scoped_iteration_vars.add(self.quantified_var)
            self.quantified_var = _subscript_value(
                self.quantified_var, suffix, scoped_iteration_vars
            )
            self.bounds = [
                bound.copy().subscript(suffix, scoped_iteration_vars)
                for bound in self.bounds
            ]
            self.premise.subscript(suffix, scoped_iteration_vars)
        return self

    def get_sympy_expression(self):
        bounds, known_funcs = AndPremise(
            self.bounds
        ).get_sympy_expression()
        body, body_funcs = self.premise.get_sympy_expression()
        known_funcs.update(body_funcs)
        if not hasattr(sympy, "ForAll"):
            raise NotImplementedError("The installed SymPy does not expose quantified formulas.")
        return sympy.ForAll(
            sympy.Symbol(self.quantified_var),
            sympy.Implies(bounds, body),
        ), known_funcs

    def read_accesses(self):
        reads = []
        for bound in self.bounds:
            reads.extend(bound.read_accesses())
        reads.extend(self.premise.read_accesses())
        return reads

    def __repr__(self):
        bounds = " and ".join(map(str, self.bounds))
        return "forall " + self.quantified_var + ": " + bounds + " => " + str(self.premise)

    def __str__(self):
        return self.__repr__()


ConditionConstraint = ExpressionPremise
AllOfConstraint = AndPremise
NotConstraint = NotPremise


class Schedule:

    def __init__(self, schedule):
        self.schedule = schedule

    def occurs_in(self, iter_var: str) -> bool:
        return iter_var in self.schedule

    def __str__(self):
        return '(' + ', '.join(self.schedule) + ')'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __copy__(self):
        new_sched = [s for s in self.schedule]
        return Schedule(new_sched)


class DataDependence:
    def __init__(
            self,
            from_si,
            to_si,
            data,
            initial_premises,
            type,
            from_domain=None,
            to_domain=None,
            from_statement_id=None,
            to_statement_id=None,
    ):
        self.from_si = from_si
        self.to_si = to_si
        self.from_statement_id = from_statement_id
        self.to_statement_id = to_statement_id
        self.kind = "ordering"
        # Keep these even if update_dep later truncates the schedules. Inner-loop
        # iterators can still occur in access expressions and domain premises.
        self.from_iteration_vars = {
            value for value in from_si.schedule if not value.isnumeric()
        }
        self.to_iteration_vars = {
            value for value in to_si.schedule if not value.isnumeric()
        }
        self.data = data
        # self.from_ind = from_ind
        # self.to_ind = to_ind
        self.premises = initial_premises
        self.from_domain = from_domain or []
        self.to_domain = to_domain or []
        self.type = type
        # assert len(from_ind) == len(to_ind)

    def add_premises(self, premises):
        self.premises.extend(premises)

    def has_premise(self, condition):
        return condition in self.premises

    def all_premises(self):
        return self.from_domain + self.to_domain + self.premises

    def get_condition_statement_sympy(self):
        condition = sympy.true
        known_funcs = set()
        for premise in self.all_premises():
            premise_expr, premise_known_funcs = premise.get_sympy_expression()
            known_funcs.update(premise_known_funcs)
            condition = sympy.And(premise_expr, condition)
        return condition, known_funcs

    def __repr__(self):
        condition = ' and '.join(map(str, self.all_premises()))
        return str(self.from_si) + '->' + str(
            self.to_si) + '\ndata: `' + self.data.__repr__() + '`\ncondition: ' + condition + '\ntype: ' + self.type
