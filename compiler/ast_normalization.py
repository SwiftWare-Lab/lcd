"""Semantics-aware normalization of simple local aliases."""

import ast
from copy import deepcopy


def normalize_aliases(module):
    """Expand removable scalar aliases in a function AST.

    The pass handles side-effect-free expressions composed of names, array
    accesses, and arithmetic. Named size definitions such as
    ``n = len(indptr) - 1`` are deliberately retained: later stages lift
    those names into host-computed scalar parameters.

    An alias is considered only when its target has one syntactic assignment
    in the function and is not a parameter or loop iterator.

    Array writes invalidate aliases that read that array. Assignments are
    removed only when no unsubstituted load of the alias remains, so a value
    that must retain snapshot semantics stays in the normalized program.
    """
    normalized = deepcopy(module)
    functions = [
        node
        for node in normalized.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for function in functions:
        _FunctionAliasNormalizer(function).normalize()
    return ast.fix_missing_locations(normalized)


class _FunctionAliasNormalizer:
    def __init__(self, function):
        self.function = function
        self.assignment_counts = _assignment_counts(function)
        self.parameters = {
            argument.arg
            for argument in (
                list(function.args.posonlyargs)
                + list(function.args.args)
                + list(function.args.kwonlyargs)
            )
        }
        if function.args.vararg is not None:
            self.parameters.add(function.args.vararg.arg)
        if function.args.kwarg is not None:
            self.parameters.add(function.args.kwarg.arg)
        self.loop_iterators = {
            node.target.id
            for node in ast.walk(function)
            if (
                isinstance(node, (ast.For, ast.AsyncFor))
                and isinstance(node.target, ast.Name)
            )
        }
        self.alias_assignments = set()

    def normalize(self):
        self.function.body = self._statements(self.function.body, {})
        remaining_loads = {
            node.id
            for node in ast.walk(self.function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        removable = self.alias_assignments - remaining_loads
        self.function.body = _remove_alias_assignments(
            self.function.body,
            removable,
        )

    def _statements(self, statements, aliases):
        result = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                result.append(self._assign(statement, aliases))
            elif isinstance(statement, ast.AnnAssign):
                result.append(self._ann_assign(statement, aliases))
            elif isinstance(statement, ast.AugAssign):
                result.append(self._aug_assign(statement, aliases))
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                result.append(self._for(statement, aliases))
            elif isinstance(statement, ast.If):
                result.append(self._if(statement, aliases))
            elif isinstance(statement, ast.While):
                result.append(self._while(statement, aliases))
            elif isinstance(statement, ast.Return):
                statement.value = self._expression(statement.value, aliases)
                result.append(statement)
            elif isinstance(statement, ast.Expr):
                statement.value = self._expression(statement.value, aliases)
                result.append(statement)
            else:
                result.append(_substitute(statement, aliases))
                self._invalidate(aliases, _stored_roots(statement))
        return result

    def _assign(self, statement, aliases):
        statement.value = self._expression(statement.value, aliases)
        statement.targets = [
            self._target(target, aliases)
            for target in statement.targets
        ]
        stored = set().union(*(
            _target_roots(target) for target in statement.targets
        ))
        self._invalidate(aliases, stored)

        if len(statement.targets) != 1:
            return statement
        target = statement.targets[0]
        if not self._is_alias_candidate(target, statement.value):
            return statement

        aliases[target.id] = _Alias(
            deepcopy(statement.value),
            _loaded_names(statement.value),
        )
        self.alias_assignments.add(target.id)
        return statement

    def _ann_assign(self, statement, aliases):
        statement.value = self._expression(statement.value, aliases)
        statement.target = self._target(statement.target, aliases)
        stored = _target_roots(statement.target)
        self._invalidate(aliases, stored)
        if self._is_alias_candidate(statement.target, statement.value):
            aliases[statement.target.id] = _Alias(
                deepcopy(statement.value),
                _loaded_names(statement.value),
            )
            self.alias_assignments.add(statement.target.id)
        return statement

    def _aug_assign(self, statement, aliases):
        statement.value = self._expression(statement.value, aliases)
        statement.target = self._target(statement.target, aliases)
        self._invalidate(aliases, _target_roots(statement.target))
        return statement

    def _for(self, statement, aliases):
        statement.iter = self._expression(statement.iter, aliases)
        statement.target = self._target(statement.target, aliases)
        body_aliases = dict(aliases)
        self._invalidate(body_aliases, _target_roots(statement.target))
        statement.body = self._statements(statement.body, body_aliases)
        statement.orelse = self._statements(
            statement.orelse,
            dict(aliases),
        )
        self._invalidate(
            aliases,
            _stored_roots_from_statements(statement.body + statement.orelse),
        )
        return statement

    def _if(self, statement, aliases):
        statement.test = self._expression(statement.test, aliases)
        statement.body = self._statements(statement.body, dict(aliases))
        statement.orelse = self._statements(statement.orelse, dict(aliases))
        self._invalidate(
            aliases,
            _stored_roots_from_statements(statement.body + statement.orelse),
        )
        return statement

    def _while(self, statement, aliases):
        statement.test = self._expression(statement.test, aliases)
        statement.body = self._statements(statement.body, dict(aliases))
        statement.orelse = self._statements(statement.orelse, dict(aliases))
        self._invalidate(
            aliases,
            _stored_roots_from_statements(statement.body + statement.orelse),
        )
        return statement

    @staticmethod
    def _expression(expression, aliases):
        if expression is None:
            return None
        return _substitute(expression, aliases)

    @staticmethod
    def _target(target, aliases):
        if isinstance(target, ast.Subscript):
            target.value = _substitute(target.value, aliases)
            target.slice = _substitute(target.slice, aliases)
        elif isinstance(target, (ast.Tuple, ast.List)):
            target.elts = [
                _FunctionAliasNormalizer._target(element, aliases)
                for element in target.elts
            ]
        return target

    def _is_alias_candidate(self, target, value):
        return bool(
            isinstance(target, ast.Name)
            and self.assignment_counts.get(target.id) == 1
            and target.id not in self.parameters
            and target.id not in self.loop_iterators
            and target.id not in _loaded_names(value)
            and _is_pure_alias_expression(value)
            and not _contains_size_query(value)
            and not isinstance(value, ast.Constant)
        )

    @staticmethod
    def _invalidate(aliases, stored_names):
        invalid = set(stored_names)
        changed = True
        while changed:
            changed = False
            for name, alias in list(aliases.items()):
                if name in invalid or alias.dependencies & invalid:
                    invalid.add(name)
                    aliases.pop(name, None)
                    changed = True


class _Alias:
    def __init__(self, expression, dependencies):
        self.expression = expression
        self.dependencies = set(dependencies)


class _AliasSubstituter(ast.NodeTransformer):
    def __init__(self, aliases):
        self.aliases = aliases

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.aliases:
            replacement = deepcopy(self.aliases[node.id].expression)
            return ast.copy_location(replacement, node)
        return node


def _substitute(node, aliases):
    return _AliasSubstituter(aliases).visit(deepcopy(node))


def _is_pure_alias_expression(node):
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return _is_pure_alias_expression(node.value)
    if isinstance(node, ast.Subscript):
        return (
            _is_pure_alias_expression(node.value)
            and _is_pure_alias_expression(node.slice)
        )
    if isinstance(node, ast.BinOp):
        return (
            _is_pure_alias_expression(node.left)
            and _is_pure_alias_expression(node.right)
        )
    if isinstance(node, ast.UnaryOp):
        return _is_pure_alias_expression(node.operand)
    if isinstance(node, ast.Call):
        return bool(
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and _is_pure_alias_expression(node.args[0])
        )
    return False


def _contains_size_query(node):
    """Return whether an expression derives an array size at runtime."""
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "len"
        for child in ast.walk(node)
    )


def _assignment_counts(function):
    counts = {}
    for node in ast.walk(function):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            for name in _stored_names_in_target(target):
                counts[name] = counts.get(name, 0) + 1
    return counts


def _stored_names_in_target(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(
            _stored_names_in_target(element) for element in target.elts
        )) if target.elts else set()
    return set()


def _loaded_names(node):
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _target_roots(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Subscript):
        root = target.value
        while isinstance(root, (ast.Subscript, ast.Attribute)):
            root = root.value
        return {root.id} if isinstance(root, ast.Name) else set()
    if isinstance(target, ast.Attribute):
        root = target
        while isinstance(root, ast.Attribute):
            root = root.value
        return {root.id} if isinstance(root, ast.Name) else set()
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(
            _target_roots(element) for element in target.elts
        )) if target.elts else set()
    return set()


def _stored_roots(node):
    roots = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            roots.add(child.id)
        elif isinstance(child, (ast.Subscript, ast.Attribute)) and isinstance(
                child.ctx,
                ast.Store,
        ):
            roots.update(_target_roots(child))
    return roots


def _stored_roots_from_statements(statements):
    roots = set()
    for statement in statements:
        roots.update(_stored_roots(statement))
    return roots


def _remove_alias_assignments(statements, removable):
    result = []
    for statement in statements:
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            statement.body = _remove_alias_assignments(
                statement.body,
                removable,
            ) or [ast.copy_location(ast.Pass(), statement)]
            statement.orelse = _remove_alias_assignments(
                statement.orelse,
                removable,
            )
        elif isinstance(statement, ast.If):
            statement.body = _remove_alias_assignments(
                statement.body,
                removable,
            ) or [ast.copy_location(ast.Pass(), statement)]
            statement.orelse = _remove_alias_assignments(
                statement.orelse,
                removable,
            )

        target = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
        if isinstance(target, ast.Name) and target.id in removable:
            continue
        result.append(statement)
    return result
