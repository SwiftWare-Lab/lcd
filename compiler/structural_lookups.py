"""Hoist reusable structure-only sparse searches into the host inspector.

The first supported relation is a self lookup in one pointer-delimited
segment::

    for position in range(pointer[segment], pointer[segment + 1]):
        if indices[position] == segment:
            ...
            break

The inspector materializes the matching position for every segment.  CUDA
lowering then replaces the search with a table lookup while leaving numerical
array loads in the executor.  This is the diagonal-position pattern used by
both CSR and CSC ILU0, but neither the analysis nor the rewrite names ILU0.
"""

import ast
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


SEGMENTED_MONOTONIC = "segmented_monotonically_increasing"
LOOKUP_NAME_PREFIX = "_lcd_lookup_position_"


class StructuralLookupError(Exception):
    """Raised when concrete inputs violate a structural lookup plan."""


@dataclass(frozen=True)
class StructuralLookup:
    """One position table derived solely from sparse structure arrays."""

    name: str
    pointer_array: str
    index_array: str
    sorted_segments: bool = False

    @property
    def argument_names(self):
        return self.pointer_array, self.index_array


@dataclass(frozen=True)
class _LookupMatch:
    iterator: str
    segment: ast.expr
    matched_body: tuple[ast.stmt, ...]
    pointer_array: str
    index_array: str


def create_structural_lookup_plan(function_ast, function_props=None):
    """Find reusable canonical structural searches in normalized code."""
    array_pairs = _segmented_array_pairs(function_props)
    if not array_pairs:
        return ()

    used_names = {
        node.id
        for node in ast.walk(function_ast)
        if isinstance(node, ast.Name)
    }
    planned_pairs = []
    for node in ast.walk(function_ast):
        if not isinstance(node, ast.For):
            continue
        match = _match_self_segment_lookup(node, array_pairs)
        if match is None:
            continue
        pair = match.pointer_array, match.index_array
        if pair not in planned_pairs:
            planned_pairs.append(pair)

    plan = []
    for position, (pointer_array, index_array) in enumerate(planned_pairs):
        suffix = position
        name = f"{LOOKUP_NAME_PREFIX}{suffix}"
        while name in used_names:
            suffix += 1
            name = f"{LOOKUP_NAME_PREFIX}{suffix}"
        used_names.add(name)
        plan.append(StructuralLookup(
            name=name,
            pointer_array=pointer_array,
            index_array=index_array,
            sorted_segments=True,
        ))
    return tuple(plan)


def structural_argument_names(plan):
    """Return structure inputs that must participate in cache fingerprints."""
    names = []
    for lookup in plan or ():
        for name in lookup.argument_names:
            if name not in names:
                names.append(name)
    return names


def structural_lookup_names(plan):
    return [lookup.name for lookup in plan or ()]


def materialize_structural_lookups(plan, argument_values):
    """Build all lookup arrays for one concrete sparsity structure."""
    values = {}
    for lookup in plan or ():
        try:
            pointers = np.asarray(argument_values[lookup.pointer_array])
            indices = np.asarray(argument_values[lookup.index_array])
        except KeyError as error:
            raise StructuralLookupError(
                f"Missing structural argument {error.args[0]!r}."
            ) from error
        values[lookup.name] = _materialize_lookup(
            lookup,
            pointers,
            indices,
        )
    return values


def rewrite_structural_lookups(function_ast, plan, function_props=None):
    """Replace planned searches with accesses to inspector-produced tables."""
    if not plan:
        return deepcopy(function_ast)
    array_pairs = _segmented_array_pairs(function_props)
    lookup_by_pair = {
        (lookup.pointer_array, lookup.index_array): lookup
        for lookup in plan
    }

    class LookupRewriter(ast.NodeTransformer):
        def visit_For(self, node):
            node = self.generic_visit(node)
            match = _match_self_segment_lookup(node, array_pairs)
            if match is None:
                return node
            lookup = lookup_by_pair.get(
                (match.pointer_array, match.index_array)
            )
            if lookup is None:
                return node
            position = ast.Assign(
                targets=[ast.Name(id=match.iterator, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id=lookup.name, ctx=ast.Load()),
                    slice=deepcopy(match.segment),
                    ctx=ast.Load(),
                ),
            )
            matched = ast.If(
                test=ast.Compare(
                    left=ast.Name(id=match.iterator, ctx=ast.Load()),
                    ops=[ast.GtE()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[deepcopy(statement) for statement in match.matched_body],
                orelse=[],
            )
            return [
                ast.copy_location(position, node),
                ast.copy_location(matched, node),
            ]

    rewritten = LookupRewriter().visit(deepcopy(function_ast))
    return ast.fix_missing_locations(rewritten)


def _segmented_array_pairs(function_props):
    pairs = set()
    for key, properties in (function_props or {}).items():
        if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(name, str) for name in key)
        ):
            continue
        if any(_property_name(property_) == SEGMENTED_MONOTONIC
               for property_ in properties):
            pairs.add((key[0], key[1]))
    return pairs


def _property_name(property_):
    if isinstance(property_, (tuple, list)) and property_:
        return property_[0]
    return property_


def _match_self_segment_lookup(node, array_pairs):
    if (
            node.orelse
            or not isinstance(node.target, ast.Name)
            or not isinstance(node.iter, ast.Call)
            or not isinstance(node.iter.func, ast.Name)
            or node.iter.func.id != "range"
            or len(node.iter.args) != 2
            or node.iter.keywords
            or len(node.body) != 1
            or not isinstance(node.body[0], ast.If)
    ):
        return None

    condition = node.body[0]
    if (
            condition.orelse
            or not condition.body
            or not isinstance(condition.body[-1], ast.Break)
            or not isinstance(condition.test, ast.Compare)
            or len(condition.test.ops) != 1
            or not isinstance(condition.test.ops[0], ast.Eq)
            or len(condition.test.comparators) != 1
    ):
        return None

    iterator = node.target.id
    lower, upper = node.iter.args
    left = condition.test.left
    right = condition.test.comparators[0]
    for pointer_array, index_array in array_pairs:
        segment = _consecutive_segment(lower, upper, pointer_array)
        if segment is None:
            continue
        if _is_index_lookup(left, index_array, iterator):
            needle = right
        elif _is_index_lookup(right, index_array, iterator):
            needle = left
        else:
            continue
        if ast.dump(needle) != ast.dump(segment):
            continue
        return _LookupMatch(
            iterator=iterator,
            segment=deepcopy(segment),
            matched_body=tuple(deepcopy(condition.body[:-1])),
            pointer_array=pointer_array,
            index_array=index_array,
        )
    return None


def _consecutive_segment(lower, upper, pointer_array):
    if not (
            _is_subscript_of(lower, pointer_array)
            and _is_subscript_of(upper, pointer_array)
    ):
        return None
    expected_upper = ast.BinOp(
        left=deepcopy(lower.slice),
        op=ast.Add(),
        right=ast.Constant(value=1),
    )
    if ast.dump(upper.slice) != ast.dump(expected_upper):
        return None
    return lower.slice


def _is_subscript_of(node, array_name):
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == array_name
    )


def _is_index_lookup(node, array_name, iterator):
    return (
        _is_subscript_of(node, array_name)
        and isinstance(node.slice, ast.Name)
        and node.slice.id == iterator
    )


def _materialize_lookup(lookup, pointers, indices):
    if pointers.ndim != 1 or indices.ndim != 1:
        raise StructuralLookupError(
            f"{lookup.pointer_array} and {lookup.index_array} must be 1-D."
        )
    if pointers.size == 0:
        raise StructuralLookupError(
            f"{lookup.pointer_array} must contain at least one offset."
        )
    if not np.issubdtype(pointers.dtype, np.integer):
        raise StructuralLookupError(
            f"{lookup.pointer_array} must contain integer offsets."
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise StructuralLookupError(
            f"{lookup.index_array} must contain integer coordinates."
        )

    node_count = pointers.size - 1
    positions = np.full(node_count, -1, dtype=np.int32)
    previous = int(pointers[0])
    if previous < 0 or previous > indices.size:
        raise StructuralLookupError(
            f"{lookup.pointer_array}[0] is outside {lookup.index_array}."
        )

    for segment in range(node_count):
        start = int(pointers[segment])
        end = int(pointers[segment + 1])
        if start < previous or end < start or end > indices.size:
            raise StructuralLookupError(
                f"Invalid pointer segment {segment} in "
                f"{lookup.pointer_array}."
            )
        coordinates = indices[start:end]
        if (
                lookup.sorted_segments
                and coordinates.size > 1
                and np.any(coordinates[1:] < coordinates[:-1])
        ):
            raise StructuralLookupError(
                f"Segment {segment} of {lookup.index_array} violates the "
                "segmented_monotonically_increasing property."
            )
        if lookup.sorted_segments:
            relative = int(np.searchsorted(coordinates, segment, side="left"))
            position = start + relative
            if position < end and int(indices[position]) == segment:
                positions[segment] = position
        else:
            matches = np.flatnonzero(coordinates == segment)
            if matches.size:
                positions[segment] = start + int(matches[0])
        previous = end
    return positions


__all__ = [
    "StructuralLookup",
    "StructuralLookupError",
    "create_structural_lookup_plan",
    "materialize_structural_lookups",
    "rewrite_structural_lookups",
    "structural_argument_names",
    "structural_lookup_names",
]
