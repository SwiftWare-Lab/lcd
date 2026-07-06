"""Generate runtime DAG builders from ordering dependences.

This module intentionally does not import legality_check.  Its input is the
already-filtered list of ordering dependences plus SPF node data.
"""

import ast
from dataclasses import dataclass, field

from compiler.spf_def import (
    AndPremise,
    DependencePremise,
    Equality,
    ExpressionPremise,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    NotEqual,
    NotPremise,
    OrPremise,
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


@dataclass
class EdgeRule:
    iterators: list[IteratorSpec]
    source: object
    destination: object
    guards: list[object] = field(default_factory=list)
    name_replacements: dict[str, str] = field(default_factory=dict)
    bindings: dict[str, object] = field(default_factory=dict)
    dependence: object = None


@dataclass
class DAGPlan:
    arguments: list[str]
    node_count: object
    edge_rules: list[EdgeRule]
    parallel_iterator: str


class DAGPlanner:
    """Translate functional dependence relations into runtime edge rules."""

    def __init__(self, node_data, parallel_iterator, arguments):
        self.node_data = node_data
        self.parallel_iterator = parallel_iterator
        self.arguments = list(arguments)
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
        edge_rules = [
            self._create_edge_rule(dependence)
            for dependence in ordering_dependences
        ]
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

        binding_premise, source, destination, enumeration_side, bindings = (
            self._solve_endpoints(
                dependence,
                source_name,
                destination_name,
            )
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

        guards = [
            premise
            for premise in dependence.premises
            if premise is not binding_premise
        ]
        if source_statement.domain:
            guards.extend(
                dependence.from_domain[-len(source_statement.domain):]
            )
        if destination_statement.domain:
            guards.extend(
                dependence.to_domain[-len(destination_statement.domain):]
            )
        return EdgeRule(
            iterators=iterators,
            source=source,
            destination=destination,
            guards=guards,
            name_replacements=name_replacements,
            bindings=bindings,
            dependence=dependence,
        )

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

        raise DAGCodegenError(
            "The dependence is not functional in either task endpoint. "
            "Expected an equality that binds the source or destination "
            f"iterator ({source_name!r}, {destination_name!r})."
        )

    def _statement_iterators(self, statement):
        iterators = []
        for schedule_component in statement.schedule.schedule:
            if schedule_component.isnumeric():
                continue
            loop_node = self.loop_nodes.get(schedule_component)
            if loop_node is None:
                raise DAGCodegenError(
                    f"No loop metadata exists for {schedule_component!r}."
                )
            iterators.append(
                IteratorSpec(
                    name=schedule_component,
                    lower=loop_node.bound.start,
                    upper=loop_node.bound.end,
                )
            )
        return iterators


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
            return ast.BoolOp(
                op=ast.And(),
                values=[self.premise(child) for child in premise.premises],
            )
        if isinstance(premise, OrPremise):
            return ast.BoolOp(
                op=ast.Or(),
                values=[self.premise(child) for child in premise.premises],
            )
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
                dependence_guards,
        ):
            guards.append(ast.Compare(
                left=source,
                ops=[ast.NotEq()],
                comparators=[destination],
            ))
        guards.extend(dependence_guards)
        guard_expression = (
            guards[0]
            if len(guards) == 1
            else ast.BoolOp(op=ast.And(), values=guards)
        )
        statements.append(
            ast.If(
                test=guard_expression,
                body=[
                    ast.Expr(
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
                ],
                orelse=[],
            )
        )

        nested = statements
        for iterator in reversed(edge_rule.iterators):
            nested = [
                ast.For(
                    target=ast.Name(id=iterator.name, ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id="range", ctx=ast.Load()),
                        args=[
                            _ExpressionLowerer().value(iterator.lower),
                            _ExpressionLowerer().value(iterator.upper),
                        ],
                        keywords=[],
                    ),
                    body=nested,
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
):
    return DAGPlanner(
        node_data=node_data,
        parallel_iterator=parallel_iterator,
        arguments=arguments,
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
