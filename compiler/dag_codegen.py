"""Generate runtime DAG builders from ordering dependences.

This module intentionally does not import legality_check.  Its input is the
already-filtered list of ordering dependences plus SPF node data.
"""

import ast
from dataclasses import dataclass, field
import re

from compiler.spf_def import (
    AndPremise,
    DependencePremise,
    Equality,
    ExpressionPremise,
    ForAllPremise,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    NotEqual,
    NotPremise,
    OrPremise,
    SubstitutePremise,
)
from compiler.spf_raising import Access, AssignNode, Expression, LoopNode


class DAGCodegenError(Exception):
    """Raised when a dependence cannot be lowered to an efficient edge rule."""


EXECUTION_MODES = frozenset({"sync_free", "wavefront"})


@dataclass(frozen=True)
class IteratorSpec:
    name: str
    lower: object
    upper: object
    break_before: tuple = ()
    break_after: tuple = ()


@dataclass(frozen=True)
class OwnershipFact:
    """A position belongs to a half-open region of a boundary array.

    ``boundary[owner] <= position < boundary[owner + 1]``
    """

    boundary: object
    position: object
    owner: object


@dataclass
class EdgeRule:
    iterators: list[IteratorSpec]
    source: object
    destination: object
    guards: list[object] = field(default_factory=list)
    name_replacements: dict[str, str] = field(default_factory=dict)
    bindings: dict[str, object] = field(default_factory=dict)
    dependence: object = None
    structural: bool = False


@dataclass
class DAGPlan:
    arguments: list[str]
    node_count: object
    edge_rules: list[EdgeRule]
    parallel_iterator: str


class DAGPlanner:
    """Translate dependence relations into runtime edge-enumeration rules."""

    def __init__(
            self,
            node_data,
            parallel_iterator,
            arguments,
            function_props=None,
    ):
        self.node_data = node_data
        self.parallel_iterator = parallel_iterator
        self.arguments = list(arguments)
        self.function_props = function_props or {}
        self.loop_nodes = {
            node.iter_var: node
            for node in node_data.values()
            if isinstance(node, LoopNode)
        }
        if parallel_iterator not in self.loop_nodes:
            raise DAGCodegenError(
                f"Unknown parallel iterator {parallel_iterator!r}."
            )

    def build(self, ordering_dependences):
        edge_rules = []
        structural_keys = set()
        for dependence in ordering_dependences:
            edge_rule = self._create_edge_rule(dependence)
            if edge_rule.structural:
                key = self._edge_rule_key(edge_rule)
                if key in structural_keys:
                    continue
                structural_keys.add(key)
            edge_rules.append(edge_rule)
        return DAGPlan(
            arguments=self.arguments,
            node_count=self.loop_nodes[self.parallel_iterator].bound.end,
            edge_rules=edge_rules,
            parallel_iterator=self.parallel_iterator,
        )

    def _create_edge_rule(self, dependence):
        if getattr(dependence, "kind", "ordering") != "ordering":
            raise DAGCodegenError(
                "Reduction dependences must not be passed to DAG generation."
            )

        source_statement = self._statement(dependence.from_statement_id)
        destination_statement = self._statement(dependence.to_statement_id)
        source_name = self.parallel_iterator + "1"
        destination_name = self.parallel_iterator + "2"

        endpoint_solution = self._solve_endpoints(
            dependence,
            source_name,
            destination_name,
        )
        if endpoint_solution is None:
            structural_solution = self._solve_structural_endpoints(
                dependence,
                source_statement,
                destination_statement,
                source_name,
                destination_name,
            )
            if structural_solution is not None:
                return self._create_structural_edge_rule(
                    dependence,
                    source_statement,
                    destination_statement,
                    source_name,
                    destination_name,
                    structural_solution,
                )
            return self._create_relational_edge_rule(
                dependence,
                source_statement,
                destination_statement,
                source_name,
                destination_name,
            )

        binding_premise, source, destination, enumeration_side, bindings = (
            endpoint_solution
        )

        enumerated_statement = (
            source_statement
            if enumeration_side == 1
            else destination_statement
        )
        iterators = self._statement_iterators(enumerated_statement)
        suffix = str(enumeration_side)
        name_replacements = {
            iterator + suffix: iterator
            for iterator in (
                dependence.from_iteration_vars
                if enumeration_side == 1
                else dependence.to_iteration_vars
            )
        }

        guards = self._lowerable_guards(
            [
                premise
                for premise in dependence.premises
                if premise is not binding_premise
            ],
            dependence,
            name_replacements,
            bindings,
        )
        if source_statement.domain:
            guards.extend(
                dependence.from_domain[-len(source_statement.domain):]
            )
        if destination_statement.domain:
            guards.extend(
                dependence.to_domain[-len(destination_statement.domain):]
            )
        available_names = (
            set(self.arguments)
            | set(name_replacements)
            | set(bindings)
            | {iterator.name for iterator in iterators}
        )
        guards = self._filter_inspector_guards(guards, available_names)
        return EdgeRule(
            iterators=iterators,
            source=source,
            destination=destination,
            guards=guards,
            name_replacements=name_replacements,
            bindings=bindings,
            dependence=dependence,
        )

    def _create_structural_edge_rule(
            self,
            dependence,
            source_statement,
            destination_statement,
            source_name,
            destination_name,
            endpoint_solution,
    ):
        """Project a relation onto one statement traversal.

        Ownership/equality reasoning proves the endpoint binding. Conditions
        unavailable in the selected traversal are existentially projected
        out, so this is a conservative DAG over-approximation.
        """
        source, destination, enumeration_side, bindings = endpoint_solution
        statement = (
            source_statement
            if enumeration_side == 1
            else destination_statement
        )
        suffix = str(enumeration_side)
        side_iteration_vars = (
            dependence.from_iteration_vars
            if enumeration_side == 1
            else dependence.to_iteration_vars
        )
        required_names = self._required_iterator_bases(
            (source, destination, *bindings.values()),
            side_iteration_vars,
            suffix,
        )
        required_names.add(self.parallel_iterator)
        iterators = self._statement_iterators(
            statement,
            required_names=required_names,
        )
        enumerated_names = {iterator.name for iterator in iterators}
        name_replacements = {
            name + suffix: name
            for name in side_iteration_vars
            if name in enumerated_names
        }

        # Endpoint ordering remains relevant. Pointer equalities have already
        # been consumed by the proof, and retaining them would reintroduce the
        # iteration space that was just projected out.
        guards = [
            premise
            for premise in dependence.premises
            if not isinstance(premise, Equality)
        ]
        copied_domain = (
            dependence.from_domain
            if enumeration_side == 1
            else dependence.to_domain
        )
        if statement.domain:
            guards.extend(copied_domain[-len(statement.domain):])
        available_names = (
            set(self.arguments)
            | set(name_replacements)
            | set(bindings)
        )
        guards = self._filter_inspector_guards(guards, available_names)
        guards = self._lowerable_guards(
            guards,
            dependence,
            name_replacements,
            bindings,
        )
        return EdgeRule(
            iterators=iterators,
            source=source,
            destination=destination,
            guards=guards,
            name_replacements=name_replacements,
            bindings=bindings,
            dependence=dependence,
            structural=True,
        )

    def _create_relational_edge_rule(
            self,
            dependence,
            source_statement,
            destination_statement,
            source_name,
            destination_name,
    ):
        """Enumerate both statement spaces for a general dependence relation."""
        source_iterators = self._statement_iterators(
            source_statement,
            suffix="1",
        )
        destination_iterators = self._statement_iterators(
            destination_statement,
            suffix="2",
        )
        iterators = source_iterators + destination_iterators
        copied_iterator_names = {
            iterator.name
            for iterator in iterators
        }
        name_replacements = {
            name: name
            for name in copied_iterator_names
        }

        guards = list(dependence.premises)
        if source_statement.domain:
            guards.extend(
                dependence.from_domain[-len(source_statement.domain):]
            )
        if destination_statement.domain:
            guards.extend(
                dependence.to_domain[-len(destination_statement.domain):]
            )

        # Quantified prefix conditions are represented by IteratorSpec break
        # metadata. Guards involving scalar temporaries unavailable to the
        # inspector are conservatively omitted, producing a safe DAG
        # over-approximation instead of invalid generated Python.
        available_names = set(self.arguments) | copied_iterator_names
        guards = self._filter_inspector_guards(guards, available_names)
        guards = self._lowerable_guards(
            guards,
            dependence,
            name_replacements,
            {},
        )
        return EdgeRule(
            iterators=iterators,
            source=source_name,
            destination=destination_name,
            guards=guards,
            name_replacements=name_replacements,
            dependence=dependence,
        )

    @staticmethod
    def _filter_inspector_guards(guards, available_names):
        return [
            guard
            for guard in guards
            if not isinstance(guard, (ForAllPremise, SubstitutePremise))
            and _symbolic_names(guard) <= available_names
        ]

    def _lowerable_guards(
            self,
            guards,
            dependence,
            name_replacements,
            bindings,
    ):
        lowerable_guards = []
        for guard in guards:
            unresolved = self._unresolved_copied_iterators(
                guard,
                dependence,
                name_replacements,
                bindings,
            )
            if not unresolved:
                lowerable_guards.append(guard)
                continue
            if all(
                    self._copied_iterator_base(name)
                    != self.parallel_iterator
                    for name in unresolved
            ):
                # These are ordering constraints for enclosing/host loops
                # such as ``sweep2 > sweep1``.  The generated launcher keeps
                # those loops sequential, so the DAG inspector should only
                # describe dependences between block-mapped tasks.
                continue
            raise DAGCodegenError(
                "Cannot lower DAG guard "
                f"{guard!r}; unresolved copied iterator(s): "
                + ", ".join(sorted(unresolved))
            )
        return lowerable_guards

    def _unresolved_copied_iterators(
            self,
            guard,
            dependence,
            name_replacements,
            bindings,
    ):
        available = set(name_replacements) | set(bindings)
        copied_iterators = {
            iterator + "1"
            for iterator in dependence.from_iteration_vars
        } | {
            iterator + "2"
            for iterator in dependence.to_iteration_vars
        }
        return {
            name
            for name in _symbolic_names(guard)
            if name in copied_iterators and name not in available
        }

    @staticmethod
    def _copied_iterator_base(name):
        return name[:-1] if name[-1:] in {"1", "2"} else name

    def _statement(self, statement_id):
        statement = self.node_data.get(statement_id)
        if not isinstance(statement, AssignNode):
            raise DAGCodegenError(
                f"Dependence references unknown statement {statement_id!r}."
            )
        return statement

    def _solve_endpoints(self, dependence, source_name, destination_name):
        """Solve an equality for one task endpoint.

        The remaining endpoint and its statement schedule define the iteration
        space used to enumerate the edge relation.
        """
        for premise in dependence.premises:
            if not isinstance(premise, Equality):
                continue
            left = premise.left
            right = premise.right

            if left == destination_name:
                return premise, source_name, right, 1, {destination_name: right}
            if right == destination_name:
                return premise, source_name, left, 1, {destination_name: left}
            if left == source_name:
                return premise, right, destination_name, 2, {source_name: right}
            if right == source_name:
                return premise, left, destination_name, 2, {source_name: left}

        return None

    def _solve_structural_endpoints(
            self,
            dependence,
            source_statement,
            destination_statement,
            source_name,
            destination_name,
    ):
        """Infer an endpoint through segment ownership and equality closure."""
        source_ownership = self._statement_ownership(
            source_statement,
            suffix="1",
        )
        destination_ownership = self._statement_ownership(
            destination_statement,
            suffix="2",
        )
        if not source_ownership and not destination_ownership:
            return None

        equality_terms = []
        for premise in (
                list(dependence.premises)
                + list(dependence.from_domain)
                + list(dependence.to_domain)
        ):
            if isinstance(premise, Equality):
                equality_terms.append((premise.left, premise.right))

        closure = _EqualityClosure()
        for left, right in equality_terms:
            closure.union_values(left, right)
        ownership = source_ownership + destination_ownership
        for fact in ownership:
            closure.add_value(fact.position)
            closure.add_value(fact.owner)
            closure.add_value(fact.boundary)
        closure.add_value(source_name)
        closure.add_value(destination_name)

        # Equality is closed under both array access congruence and unique
        # ownership. Iterate because either inference can enable the other.
        changed = True
        while changed:
            changed = False
            values = closure.values()
            accesses = [value for value in values if isinstance(value, Access)]
            for index, left in enumerate(accesses):
                for right in accesses[index + 1:]:
                    if (
                            str(left.var) == str(right.var)
                            and closure.equivalent(left.index, right.index)
                    ):
                        changed |= closure.union_values(left, right)
            for index, left in enumerate(ownership):
                for right in ownership[index + 1:]:
                    if (
                            str(left.boundary) == str(right.boundary)
                            and closure.equivalent(
                                left.position,
                                right.position,
                            )
                    ):
                        changed |= closure.union_values(
                            left.owner,
                            right.owner,
                        )

        source_vars = {
            iterator + "1"
            for iterator in dependence.from_iteration_vars
        }
        destination_vars = {
            iterator + "2"
            for iterator in dependence.to_iteration_vars
        }
        argument_names = set(self.arguments)

        # Prefer a source endpoint expressed in destination coordinates. This
        # gives the natural row-oriented CSR inspector. If unavailable, use a
        # destination endpoint expressed in source coordinates (the CSC case).
        source_binding = closure.best_equivalent(
            source_name,
            allowed_iteration_names=destination_vars,
            argument_names=argument_names,
            iteration_order=self._copied_iteration_order(
                destination_statement,
                "2",
            ),
        )
        if source_binding is not None:
            return (
                source_binding,
                destination_name,
                2,
                {source_name: source_binding},
            )

        destination_binding = closure.best_equivalent(
            destination_name,
            allowed_iteration_names=source_vars,
            argument_names=argument_names,
            iteration_order=self._copied_iteration_order(
                source_statement,
                "1",
            ),
        )
        if destination_binding is not None:
            return (
                source_name,
                destination_binding,
                1,
                {destination_name: destination_binding},
            )
        return None

    def _copied_iteration_order(self, statement, suffix):
        reached_parallel_iterator = False
        order = []
        for component in statement.schedule.schedule:
            if component.isnumeric():
                continue
            if component == self.parallel_iterator:
                reached_parallel_iterator = True
            if reached_parallel_iterator:
                order.append(component + suffix)
        return order

    def _statement_ownership(self, statement, suffix):
        """Infer ownership solely from loop bounds and function properties."""
        facts = []
        statement_iteration_vars = {
            component
            for component in statement.schedule.schedule
            if not component.isnumeric()
        }
        reached_parallel_iterator = False
        for component in statement.schedule.schedule:
            if component.isnumeric():
                continue
            if component == self.parallel_iterator:
                reached_parallel_iterator = True
            if not reached_parallel_iterator:
                continue
            loop = self.loop_nodes.get(component)
            if loop is None:
                continue
            bound = loop.bound.subscript_iteration_vars(
                suffix,
                statement_iteration_vars,
            )
            position = component + suffix
            fact = self._ownership_from_bound(
                position,
                bound.start,
                bound.end,
                facts,
            )
            if fact is not None:
                facts.append(fact)
        return facts

    def _ownership_from_bound(self, position, lower, upper, known_facts):
        if not isinstance(upper, Access):
            return None
        boundary = upper.var
        owner = _predecessor_expression(upper.index)
        if owner is None or not self._is_monotonic_boundary(boundary):
            return None

        if (
                isinstance(lower, Access)
                and str(lower.var) == str(boundary)
                and _value_key(lower.index) == _value_key(owner)
        ):
            return OwnershipFact(boundary, position, owner)

        lower_position = _predecessor_expression(lower)
        if lower_position is None:
            return None
        for fact in known_facts:
            if (
                    str(fact.boundary) == str(boundary)
                    and _value_key(fact.position)
                    == _value_key(lower_position)
                    and _value_key(fact.owner) == _value_key(owner)
            ):
                return OwnershipFact(boundary, position, owner)
        return None

    def _is_monotonic_boundary(self, boundary):
        properties = self.function_props.get(str(boundary), ())
        return any(
            property_name == "monotonically_increasing"
            or (
                isinstance(property_name, tuple)
                and property_name
                and property_name[0] == "monotonically_increasing"
            )
            for property_name in properties
        )

    @staticmethod
    def _required_iterator_bases(values, iteration_vars, suffix):
        copied_to_base = {
            iterator + suffix: iterator
            for iterator in iteration_vars
        }
        return {
            copied_to_base[name]
            for value in values
            for name in _symbolic_names(value)
            if name in copied_to_base
        }

    @staticmethod
    def _edge_rule_key(edge_rule):
        return (
            tuple(
                (
                    iterator.name,
                    str(iterator.lower),
                    str(iterator.upper),
                    tuple(map(str, iterator.break_before)),
                    tuple(map(str, iterator.break_after)),
                )
                for iterator in edge_rule.iterators
            ),
            str(edge_rule.source),
            str(edge_rule.destination),
            tuple(map(str, edge_rule.guards)),
            tuple(sorted(edge_rule.name_replacements.items())),
            tuple(
                sorted(
                    (name, str(value))
                    for name, value in edge_rule.bindings.items()
                )
            ),
        )

    def _statement_iterators(
            self,
            statement,
            suffix=None,
            required_names=None,
    ):
        iterators = []
        reached_parallel_iterator = False
        statement_iteration_vars = {
            component
            for component in statement.schedule.schedule
            if not component.isnumeric()
        }
        for schedule_component in statement.schedule.schedule:
            if schedule_component.isnumeric():
                continue
            if schedule_component == self.parallel_iterator:
                reached_parallel_iterator = True
            if not reached_parallel_iterator:
                # Enclosing loops are host/executor loops around the
                # block-mapped kernel launches.  They are not part of the DAG
                # node space; the DAG describes only dependences among
                # instances of the block-mapped iterator and its inner loops.
                continue
            loop_node = self.loop_nodes.get(schedule_component)
            if loop_node is None:
                raise DAGCodegenError(
                    f"No loop metadata exists for {schedule_component!r}."
                )
            bound = loop_node.bound
            iterator_name = schedule_component
            break_before = []
            break_after = []
            if suffix is not None:
                bound = bound.subscript_iteration_vars(
                    suffix,
                    statement_iteration_vars,
                )
                iterator_name += suffix
            for break_condition in loop_node.break_conditions:
                copied_condition = break_condition.copy()
                if suffix is not None:
                    copied_condition.subscript(
                        suffix,
                        statement_iteration_vars,
                    )
                if self._statement_excludes_current_break(
                        statement,
                        break_condition,
                ):
                    break_before.append(copied_condition)
                else:
                    break_after.append(copied_condition)
            iterators.append(IteratorSpec(
                name=iterator_name,
                lower=bound.start,
                upper=bound.end,
                break_before=tuple(break_before),
                break_after=tuple(break_after),
            ))
            if (
                    required_names is not None
                    and required_names <= {
                        iterator.name for iterator in iterators
                    }
            ):
                break
        if not reached_parallel_iterator:
            raise DAGCodegenError(
                f"Statement {statement.id!r} is not nested under the "
                f"parallel iterator {self.parallel_iterator!r}."
            )
        return iterators

    @staticmethod
    def _statement_excludes_current_break(statement, break_condition):
        expected = str(break_condition)
        for premise in statement.domain:
            if not isinstance(premise, NotPremise):
                continue
            if str(premise.premise) == expected:
                return True
        return False


def _value_key(value):
    return (type(value).__name__, str(value))


def _predecessor_expression(value):
    """Return x for the structural expression ``x + 1``."""
    if not isinstance(value, Expression) or len(value.elements) != 3:
        return None
    left, operator, right = value.elements
    if operator != "+" or str(right) != "1":
        return None
    return left


class _EqualityClosure:
    """Small union-find over symbolic SPF values."""

    def __init__(self):
        self.parent = {}
        self.value_by_key = {}

    def add_value(self, value):
        key = _value_key(value)
        if key not in self.parent:
            self.parent[key] = key
            self.value_by_key[key] = value
        if isinstance(value, Access):
            self.add_value(value.var)
            if value.index is not None:
                self.add_value(value.index)
        elif isinstance(value, Expression):
            for element in value.elements:
                if not (
                        isinstance(element, str)
                        and element in _ExpressionLowerer.BINARY_OPERATORS
                ):
                    self.add_value(element)
        return key

    def find(self, key):
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union_values(self, left, right):
        left_key = self.add_value(left)
        right_key = self.add_value(right)
        left_root = self.find(left_key)
        right_root = self.find(right_key)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True

    def equivalent(self, left, right):
        left_key = self.add_value(left)
        right_key = self.add_value(right)
        return self.find(left_key) == self.find(right_key)

    def values(self):
        return list(self.value_by_key.values())

    def best_equivalent(
            self,
            target,
            allowed_iteration_names,
            argument_names,
            iteration_order=(),
    ):
        target_key = self.add_value(target)
        target_root = self.find(target_key)
        candidates = []
        for key, value in self.value_by_key.items():
            if key == target_key or self.find(key) != target_root:
                continue
            names = _symbolic_names(value)
            iteration_names = names - argument_names
            if not iteration_names:
                continue
            if iteration_names <= allowed_iteration_names:
                candidates.append(value)
        if not candidates:
            return None
        depth = {
            name: index
            for index, name in enumerate(iteration_order)
        }
        return min(
            candidates,
            key=lambda value: (
                max(
                    depth.get(name, len(depth))
                    for name in (
                        _symbolic_names(value) & allowed_iteration_names
                    )
                ),
                len(_symbolic_names(value)),
                len(str(value)),
                str(value),
            ),
        )


class _ExpressionLowerer:
    BINARY_OPERATORS = {
        "+": ast.Add,
        "-": ast.Sub,
        "*": ast.Mult,
        "/": ast.Div,
        "%": ast.Mod,
        "&": ast.BitAnd,
        "|": ast.BitOr,
        "^": ast.BitXor,
    }

    OPERATOR_PRECEDENCE = {
        "|": 1,
        "^": 2,
        "&": 3,
        "+": 4,
        "-": 4,
        "*": 5,
        "/": 5,
        "%": 5,
    }

    COMPARISON_OPERATORS = {
        Equality: ast.Eq,
        NotEqual: ast.NotEq,
        GreaterThan: ast.Gt,
        GreaterThanOrEqual: ast.GtE,
        LessThan: ast.Lt,
        LessThanOrEqual: ast.LtE,
    }

    def __init__(self, name_replacements=None, bindings=None):
        self.name_replacements = name_replacements or {}
        self.bindings = bindings or {}
        self._active_bindings = set()

    def value(self, value):
        if isinstance(value, Access):
            base = self.value(value.var)
            if value.index is None:
                return base
            return ast.Subscript(
                value=base,
                slice=self.value(value.index),
                ctx=ast.Load(),
            )
        if isinstance(value, Expression):
            return self._expression(value)
        if isinstance(value, (int, float, bool)):
            return ast.Constant(value=value)
        if isinstance(value, str):
            if value in self.bindings:
                if value in self._active_bindings:
                    raise DAGCodegenError(
                        f"Cyclic endpoint binding involving {value!r}."
                    )
                self._active_bindings.add(value)
                result = self.value(self.bindings[value])
                self._active_bindings.remove(value)
                return result
            replaced = self.name_replacements.get(value, value)
            if replaced.isidentifier():
                return ast.Name(id=replaced, ctx=ast.Load())
            len_match = re.fullmatch(
                r"len\(([A-Za-z_]\w*)\)",
                replaced,
            )
            if len_match:
                return ast.Call(
                    func=ast.Name(id="len", ctx=ast.Load()),
                    args=[
                        ast.Name(
                            id=len_match.group(1),
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                )
            try:
                return ast.Constant(value=ast.literal_eval(replaced))
            except (SyntaxError, ValueError):
                raise DAGCodegenError(
                    f"Cannot lower symbolic value {value!r}."
                ) from None
        raise DAGCodegenError(
            f"Unsupported DAG expression value {type(value).__name__}."
        )

    def premise(self, premise):
        if isinstance(premise, DependencePremise):
            comparison = self.COMPARISON_OPERATORS.get(type(premise))
            if comparison is None:
                raise DAGCodegenError(
                    f"Unsupported comparison {type(premise).__name__}."
                )
            return ast.Compare(
                left=self.value(premise.left),
                ops=[comparison()],
                comparators=[self.value(premise.right)],
            )
        if isinstance(premise, ExpressionPremise):
            return self.value(premise.condition)
        if isinstance(premise, AndPremise):
            values = [self.premise(child) for child in premise.premises]
            if not values:
                return ast.Constant(value=True)
            if len(values) == 1:
                return values[0]
            return ast.BoolOp(op=ast.And(), values=values)
        if isinstance(premise, OrPremise):
            values = [self.premise(child) for child in premise.premises]
            if not values:
                return ast.Constant(value=False)
            if len(values) == 1:
                return values[0]
            return ast.BoolOp(op=ast.Or(), values=values)
        if isinstance(premise, NotPremise):
            return ast.UnaryOp(
                op=ast.Not(),
                operand=self.premise(premise.premise),
            )
        raise DAGCodegenError(
            f"Unsupported guard premise {type(premise).__name__}."
        )

    def _expression(self, expression):
        elements = expression.elements
        if not elements:
            raise DAGCodegenError("Cannot lower an empty expression.")
        if len(elements) == 2 and elements[0] in {"+", "-"}:
            unary_op = ast.UAdd if elements[0] == "+" else ast.USub
            return ast.UnaryOp(op=unary_op(), operand=self.value(elements[1]))
        if len(elements) % 2 == 0:
            raise DAGCodegenError(
                f"Malformed expression {expression!r}."
            )

        values = [self.value(elements[0])]
        operators = []
        for index in range(1, len(elements), 2):
            operator = elements[index]
            if operator not in self.BINARY_OPERATORS:
                raise DAGCodegenError(
                    f"Unsupported expression operator {operator!r}."
                )
            while (
                    operators
                    and self.OPERATOR_PRECEDENCE[operators[-1]]
                    >= self.OPERATOR_PRECEDENCE[operator]
            ):
                self._apply_operator(values, operators.pop())
            operators.append(operator)
            values.append(self.value(elements[index + 1]))
        while operators:
            self._apply_operator(values, operators.pop())
        return values[0]

    def _apply_operator(self, values, operator):
        right = values.pop()
        left = values.pop()
        values.append(
            ast.BinOp(
                left=left,
                op=self.BINARY_OPERATORS[operator](),
                right=right,
            )
        )


def _symbolic_names(value):
    names = set()
    _collect_symbolic_names(value, names)
    return names


def _collect_symbolic_names(value, names):
    if isinstance(value, str):
        if value.isidentifier():
            names.add(value)
        return
    if isinstance(value, Access):
        _collect_symbolic_names(value.var, names)
        _collect_symbolic_names(value.index, names)
        return
    if isinstance(value, Expression):
        for element in value.elements:
            _collect_symbolic_names(element, names)
        return
    if isinstance(value, DependencePremise):
        _collect_symbolic_names(value.left, names)
        _collect_symbolic_names(value.right, names)
        return
    if isinstance(value, ExpressionPremise):
        _collect_symbolic_names(value.condition, names)
        return
    if isinstance(value, (AndPremise, OrPremise)):
        for premise in value.premises:
            _collect_symbolic_names(premise, names)
        return
    if isinstance(value, NotPremise):
        _collect_symbolic_names(value.premise, names)


def _normalized_comparison(predicate):
    if not (
            isinstance(predicate, ast.UnaryOp)
            and isinstance(predicate.op, ast.Not)
            and isinstance(predicate.operand, ast.Compare)
            and len(predicate.operand.ops) == 1
    ):
        return predicate
    inverse = {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Gt: ast.LtE,
        ast.GtE: ast.Lt,
        ast.Lt: ast.GtE,
        ast.LtE: ast.Gt,
    }
    comparison = predicate.operand
    inverse_operator = inverse.get(type(comparison.ops[0]))
    if inverse_operator is None:
        return predicate
    return ast.Compare(
        left=comparison.left,
        ops=[inverse_operator()],
        comparators=comparison.comparators,
    )


def _predicate_key(predicate):
    predicate = _normalized_comparison(predicate)
    if isinstance(predicate, ast.Compare) and len(predicate.ops) == 1:
        left = ast.dump(predicate.left, include_attributes=False)
        right = ast.dump(
            predicate.comparators[0],
            include_attributes=False,
        )
        operator = predicate.ops[0]
        if isinstance(operator, ast.Gt):
            return ("lt", right, left)
        if isinstance(operator, ast.Lt):
            return ("lt", left, right)
        if isinstance(operator, ast.GtE):
            return ("le", right, left)
        if isinstance(operator, ast.LtE):
            return ("le", left, right)
        if isinstance(operator, ast.Eq):
            return ("eq", *sorted((left, right)))
        if isinstance(operator, ast.NotEq):
            return ("ne", *sorted((left, right)))
    return ("ast", ast.dump(predicate, include_attributes=False))


class PythonASTEmitter:
    """Lower a DAGPlan to a Python function represented as an AST."""

    def __init__(
            self,
            validate_endpoints=False,
            execution_mode="sync_free",
    ):
        self.validate_endpoints = validate_endpoints
        self.execution_mode = _normalize_execution_mode(execution_mode)

    def generate(self, plan, function_name="extract_dag"):
        body = [self._successor_initialization(plan)]
        for index, edge_rule in enumerate(plan.edge_rules):
            body.append(self._edge_rule(plan, edge_rule, index))
        body.append(self._dependency_count_initialization(plan))
        body.append(self._dependency_count_loop())
        if self.execution_mode == "wavefront":
            body.extend(self._wavefront_statements())
        body.append(self._return_statement(self.execution_mode))

        function = ast.FunctionDef(
            name=function_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=argument) for argument in plan.arguments],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body,
            decorator_list=[],
        )
        return ast.fix_missing_locations(
            ast.Module(body=[function], type_ignores=[])
        )

    def _successor_initialization(self, plan):
        lowerer = _ExpressionLowerer()
        return ast.Assign(
            targets=[ast.Name(id="_dag_successors", ctx=ast.Store())],
            value=ast.ListComp(
                elt=ast.Call(
                    func=ast.Name(id="set", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                ),
                generators=[
                    ast.comprehension(
                        target=ast.Name(id="_dag_node", ctx=ast.Store()),
                        iter=ast.Call(
                            func=ast.Name(id="range", ctx=ast.Load()),
                            args=[lowerer.value(plan.node_count)],
                            keywords=[],
                        ),
                        ifs=[],
                        is_async=0,
                    )
                ],
            ),
        )

    def _edge_rule(self, plan, edge_rule, index):
        lowerer = _ExpressionLowerer(
            edge_rule.name_replacements,
            edge_rule.bindings,
        )
        source_name = f"_dag_source_{index}"
        destination_name = f"_dag_destination_{index}"
        statements = [
            ast.Assign(
                targets=[ast.Name(id=source_name, ctx=ast.Store())],
                value=lowerer.value(edge_rule.source),
            ),
            ast.Assign(
                targets=[ast.Name(id=destination_name, ctx=ast.Store())],
                value=lowerer.value(edge_rule.destination),
            ),
        ]

        source = ast.Name(id=source_name, ctx=ast.Load())
        destination = ast.Name(id=destination_name, ctx=ast.Load())
        dependence_guards = [
            lowerer.premise(guard)
            for guard in edge_rule.guards
        ]
        implied_guards = [
            ast.UnaryOp(
                op=ast.Not(),
                operand=lowerer.premise(condition),
            )
            for iterator in edge_rule.iterators
            for condition in iterator.break_before
        ]
        for iterator in edge_rule.iterators:
            predecessor = _predecessor_expression(iterator.lower)
            if predecessor is not None:
                implied_guards.append(ast.Compare(
                    left=ast.Name(id=iterator.name, ctx=ast.Load()),
                    ops=[ast.Gt()],
                    comparators=[lowerer.value(predecessor)],
                ))
        implied_keys = {
            _predicate_key(guard)
            for guard in implied_guards
        }
        # A native ``if condition: break`` already establishes ``not
        # condition`` for the body. Emitting it again in the edge condition
        # was both noisy and a code-generation bug.
        dependence_guards = [
            guard
            for guard in dependence_guards
            if _predicate_key(guard) not in implied_keys
        ]
        unique_guards = []
        seen_guard_keys = set()
        for guard in dependence_guards:
            key = _predicate_key(guard)
            if key in seen_guard_keys:
                continue
            seen_guard_keys.add(key)
            unique_guards.append(guard)
        dependence_guards = unique_guards
        guards = []
        if self.validate_endpoints:
            node_count = _ExpressionLowerer().value(plan.node_count)
            guards.extend([
                ast.Compare(
                    left=ast.Constant(value=0),
                    ops=[ast.LtE(), ast.Lt()],
                    comparators=[source, node_count],
                ),
                ast.Compare(
                    left=ast.Constant(value=0),
                    ops=[ast.LtE(), ast.Lt()],
                    comparators=[destination, node_count],
                ),
            ])
        if not self._has_strict_endpoint_order(
                lowerer,
                edge_rule,
                dependence_guards + implied_guards,
        ):
            guards.append(ast.Compare(
                left=source,
                ops=[ast.NotEq()],
                comparators=[destination],
            ))
        guards.extend(dependence_guards)
        add_edge = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Name(
                            id="_dag_successors",
                            ctx=ast.Load(),
                        ),
                        slice=source,
                        ctx=ast.Load(),
                    ),
                    attr="add",
                    ctx=ast.Load(),
                ),
                args=[destination],
                keywords=[],
            )
        )
        if guards:
            guard_expression = (
                guards[0]
                if len(guards) == 1
                else ast.BoolOp(op=ast.And(), values=guards)
            )
            statements.append(ast.If(
                test=guard_expression,
                body=[add_edge],
                orelse=[],
            ))
        else:
            statements.append(add_edge)

        nested = statements
        for iterator in reversed(edge_rule.iterators):
            loop_lowerer = _ExpressionLowerer(
                edge_rule.name_replacements,
                edge_rule.bindings,
            )
            loop_body = list(nested)
            for condition in reversed(iterator.break_before):
                loop_body.insert(0, ast.If(
                    test=loop_lowerer.premise(condition),
                    body=[ast.Break()],
                    orelse=[],
                ))
            for condition in iterator.break_after:
                loop_body.append(ast.If(
                    test=loop_lowerer.premise(condition),
                    body=[ast.Break()],
                    orelse=[],
                ))
            nested = [
                ast.For(
                    target=ast.Name(id=iterator.name, ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id="range", ctx=ast.Load()),
                        args=[
                            loop_lowerer.value(iterator.lower),
                            loop_lowerer.value(iterator.upper),
                        ],
                        keywords=[],
                    ),
                    body=loop_body,
                    orelse=[],
                )
            ]
        if not nested:
            raise DAGCodegenError("An edge rule has no iteration space.")
        return nested[0]

    @staticmethod
    def _has_strict_endpoint_order(lowerer, edge_rule, guards):
        source = ast.dump(lowerer.value(edge_rule.source))
        destination = ast.dump(lowerer.value(edge_rule.destination))
        for guard in guards:
            guard = _normalized_comparison(guard)
            if not isinstance(guard, ast.Compare) or len(guard.ops) != 1:
                continue
            if not isinstance(guard.ops[0], (ast.Gt, ast.Lt)):
                continue
            left = ast.dump(guard.left)
            right = ast.dump(guard.comparators[0])
            if {left, right} == {source, destination}:
                return True
        return False

    @staticmethod
    def _dependency_count_initialization(plan):
        return ast.Assign(
            targets=[ast.Name(id="dep_count", ctx=ast.Store())],
            value=ast.ListComp(
                elt=ast.Constant(value=0),
                generators=[
                    ast.comprehension(
                        target=ast.Name(
                            id="_dag_node",
                            ctx=ast.Store(),
                        ),
                        iter=ast.Call(
                            func=ast.Name(id="range", ctx=ast.Load()),
                            args=[
                                _ExpressionLowerer().value(plan.node_count)
                            ],
                            keywords=[],
                        ),
                        ifs=[],
                        is_async=0,
                    )
                ],
            ),
        )

    @staticmethod
    def _dependency_count_loop():
        return ast.For(
            target=ast.Name(id="_dag_neighbors", ctx=ast.Store()),
            iter=ast.Name(id="_dag_successors", ctx=ast.Load()),
            body=[
                ast.For(
                    target=ast.Name(
                        id="_dag_destination",
                        ctx=ast.Store(),
                    ),
                    iter=ast.Name(
                        id="_dag_neighbors",
                        ctx=ast.Load(),
                    ),
                    body=[
                        ast.AugAssign(
                            target=ast.Subscript(
                                value=ast.Name(
                                    id="dep_count",
                                    ctx=ast.Load(),
                                ),
                                slice=ast.Name(
                                    id="_dag_destination",
                                    ctx=ast.Load(),
                                ),
                                ctx=ast.Store(),
                            ),
                            op=ast.Add(),
                            value=ast.Constant(value=1),
                        )
                    ],
                    orelse=[],
                )
            ],
            orelse=[],
        )

    @staticmethod
    def _wavefront_statements():
        return ast.parse(
            """
remaining_dep_count = dep_count.copy()
wavefront_list = []
wavefront = [
    node
    for node, count in enumerate(remaining_dep_count)
    if count == 0
]
while wavefront:
    wavefront_list.append(wavefront)
    new_wavefront = set()
    for node in wavefront:
        for successor in _dag_successors[node]:
            remaining_dep_count[successor] -= 1
            if remaining_dep_count[successor] == 0:
                new_wavefront.add(successor)
    wavefront = sorted(new_wavefront)
"""
        ).body

    @staticmethod
    def _return_statement(execution_mode):
        execution_data = (
            "wavefront_list"
            if execution_mode == "wavefront"
            else "dep_count"
        )
        return ast.Return(
            value=ast.Tuple(
                elts=[
                    ast.ListComp(
                        elt=ast.Call(
                            func=ast.Name(
                                id="sorted",
                                ctx=ast.Load(),
                            ),
                            args=[
                                ast.Name(
                                    id="_dag_neighbors",
                                    ctx=ast.Load(),
                                )
                            ],
                            keywords=[],
                        ),
                        generators=[
                            ast.comprehension(
                                target=ast.Name(
                                    id="_dag_neighbors",
                                    ctx=ast.Store(),
                                ),
                                iter=ast.Name(
                                    id="_dag_successors",
                                    ctx=ast.Load(),
                                ),
                                ifs=[],
                                is_async=0,
                            )
                        ],
                    ),
                    ast.Name(id=execution_data, ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            )
        )


def create_dag_plan(
        ordering_dependences,
        node_data,
        parallel_iterator,
        arguments,
        function_props=None,
):
    return DAGPlanner(
        node_data=node_data,
        parallel_iterator=parallel_iterator,
        arguments=arguments,
        function_props=function_props,
    ).build(ordering_dependences)


def generate_dag_ast(
        plan,
        function_name="extract_dag",
        validate_endpoints=False,
        execution_mode="sync_free",
):
    return PythonASTEmitter(
        validate_endpoints=validate_endpoints,
        execution_mode=execution_mode,
    ).generate(plan, function_name=function_name)


def generate_dag_source(
        plan,
        function_name="extract_dag",
        validate_endpoints=False,
        execution_mode="sync_free",
):
    return ast.unparse(
        generate_dag_ast(
            plan,
            function_name=function_name,
            validate_endpoints=validate_endpoints,
            execution_mode=execution_mode,
        )
    )


def compile_dag_builder(
        plan,
        function_name="extract_dag",
        globals_dict=None,
        validate_endpoints=False,
        execution_mode="sync_free",
):
    namespace = dict(globals_dict or {})
    module = generate_dag_ast(
        plan,
        function_name=function_name,
        validate_endpoints=validate_endpoints,
        execution_mode=execution_mode,
    )
    print(ast.unparse(module))
    exec(compile(module, filename="<generated-dag>", mode="exec"), namespace)
    return namespace[function_name]


def _normalize_execution_mode(execution_mode):
    execution_mode = execution_mode.replace("-", "_")
    if execution_mode not in EXECUTION_MODES:
        raise DAGCodegenError(
            f"Unknown execution mode {execution_mode!r}; expected one of "
            + ", ".join(sorted(EXECUTION_MODES))
            + "."
        )
    return execution_mode
