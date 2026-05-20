import ast
from copy import deepcopy

import sympy

from compiler.lcd.lcd_def import *
from compiler.spf_def import *


# from sympy import Symbol, Eq, GreaterThan, Lt, And, Or, Not
# from copy import deepcopy
# from legality_check import check_legality_of_fusion


class LoopBound:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def subscript_iteration_vars(self, subscript):
        if isinstance(self.start, Expression):
            new_start = self.start.subscript(subscript)
        elif isinstance(self.start, Access):
            new_start = self.start.subscript_index(subscript)
        else:
            new_start = self.start
        if isinstance(self.end, Expression):
            new_end = self.end.subscript(subscript)
        elif isinstance(self.end, Access):
            new_end = self.end.subscript_index(subscript)
        else:
            new_end = self.end
        return LoopBound(new_start, new_end)

    def get_sympy_expr(self, iter_var):
        f1 = []
        f2 = []
        consts = []
        if isinstance(self.start, Access):
            f1, start = self.start.get_sympy_expr()
        elif isinstance(self.start, Expression):
            start = self.start.get_sympy_expr()
        elif self.start.isnumeric():
            start = int(self.start)
        else:
            start = sympy.Symbol(self.start)
            consts.append(start)
        if isinstance(self.end, Access):
            f2, end = self.end.get_sympy_expr()
        elif isinstance(self.end, Expression):
            end = self.end.get_sympy_expr()
        elif self.end.isnumeric():
            end = int(self.end)
        else:
            end = sympy.Symbol(self.end)
            consts.append(end)
        return consts, f1 + f2, sympy.And(
            sympy.Gt(sympy.Symbol(iter_var), start),
            sympy.LessThan(sympy.Symbol(iter_var), end),
        )

    def to_premises(self, iter_var):
        return [
            GreaterThan(iter_var, self.start),
            LessThan(iter_var, self.end),
        ]

    def __repr__(self):
        return "(" + str(self.start) + ", " + str(self.end) + ")"


class Access:
    def __init__(self, var, index):
        self.var = var
        self.index = index

    def get_sympy_expr(self):
        func = self.get_data_array()
        funcs = [func]
        index_funcs, args = self.get_index_sympy_expr()
        funcs += [f for f in index_funcs if f is not None]
        func_sym = sympy.Function(func)(args)
        return funcs, func_sym

    def subscript_index(self, subscript):
        if isinstance(self.index, Access):
            return Access(self.index.var, self.index.subscript_index(subscript))
        if isinstance(self.index, Expression):
            return Access(self.var, self.index.subscript(subscript))
        if isinstance(self.index, str) and self.index.isidentifier():
            return Access(self.var, self.index + subscript)
        return Access(self.var, self.index)

    def __eq__(self, other):
        if isinstance(other, Access):
            if type(self.index) is list and type(other.index) is list:
                self.index.sort()
                other.index.sort()
            return self.var == other.var and self.index == other.index
        return False

    def __str__(self):
        return f"{self.var}[{self.index}]" if self.index is not None else self.var

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return str(self)

    def get_data_array(self):
        if isinstance(self.var, Access):
            return self.var.get_data_array()
        return self.var

    def get_index(self):
        if isinstance(self.var, Access):
            return self.var.get_index() + (self.index,)
        return (self.index,)

    def get_index_sympy_expr(self):
        if isinstance(self.index, Access):
            return self.index.get_sympy_expr()
        if isinstance(self.index, Expression):
            return [], self.index.get_sympy_expr()
        return [], sympy.Symbol(str(self.index).replace(".", ""))


class Expression:
    def __init__(self, elements):
        self.elements = elements

    def __str__(self):
        return " ".join(list(map(str, self.elements)))

    def __repr__(self):
        return str(self)

    def get_sympy_expr(self):
        return sympy.parse_expr(str(self).replace(".", "").replace("[", "(").replace("]", ")"))

    def subscript(self, subscript):
        new_expr = Expression(deepcopy(self.elements))
        for i, element in enumerate(new_expr.elements):
            if isinstance(element, Access):
                new_expr.elements[i] = element.subscript_index(subscript)
            elif isinstance(element, str) and element.isidentifier():
                new_expr.elements[i] = element + subscript
        return new_expr


class SPFNode:
    def __init__(self, node_id):
        self.id = node_id
        self.children = []
        self.schedule = None

    def add_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return self.id

    def __str__(self):
        return self.id


class LoopNode(SPFNode):
    def __init__(self, node_id, iter_var, loop_bound):
        super().__init__(node_id)
        self.iter_var = iter_var
        self.bound = loop_bound


class AssignNode(SPFNode):
    def __init__(self, node_id, target, value):
        super().__init__(node_id)
        self.target = target
        self.value = value
        self.read_accesses = []
        self.write_access = None
        self.domain = []
        self.loop_domain = []

    def add_read_access(self, read_access):
        self.read_accesses.append(read_access)

    def add_domain_premise(self, premise):
        self.domain.append(premise)

    def __str__(self):
        return (
            self.id
            + "\n - write: "
            + str(self.write_access)
            + "\n - reads: \n"
            + "\n".join(map(str, self.read_accesses))
            + "\n - domain: \n"
            + "\n".join(map(str, self.domain))
            + "\n - loop domain: \n"
            + "\n".join(map(str, self.loop_domain))
        )


class _SPFBuilderBase:
    def __init__(self):
        self.loop_counter = 0
        self.stmt_counter = 0
        self.fused_loop_ids = []

    def next_loop_id(self):
        loop_id = "F" + str(self.loop_counter)
        self.loop_counter += 1
        return loop_id

    def next_stmt_id(self):
        stmt_id = "S" + str(self.stmt_counter)
        self.stmt_counter += 1
        return stmt_id

    def create_assign_node(self, target, value, is_augmented=False, domain=None):
        assign_node = AssignNode(self.next_stmt_id(), target, value)
        assign_node.write_access = self.convert_write_target(target)
        if is_augmented and assign_node.write_access is not None:
            assign_node.add_read_access(assign_node.write_access)
        for read_access in self.extract_read_accesses(value):
            assign_node.add_read_access(read_access)
        self.add_domain_premises(assign_node, domain or [])
        return assign_node

    def create_loop_node(self, iter_var, loop_bound, body):
        loop_node = LoopNode(self.next_loop_id(), iter_var, loop_bound)
        for stmt in body:
            loop_node.add_child(stmt)
        return loop_node

    def convert_write_target(self, target):
        raise NotImplementedError

    def extract_read_accesses(self, value):
        reads = []
        self._collect_accesses(value, reads)
        return reads

    def _collect_accesses(self, value, reads):
        if isinstance(value, Access):
            reads.append(value)
            return
        if isinstance(value, Expression):
            for element in value.elements:
                self._collect_accesses(element, reads)

    def add_domain_premises(self, assign_node, domain):
        for premise in domain:
            assign_node.add_domain_premise(premise)
            for read_access in premise.read_accesses():
                if read_access not in assign_node.read_accesses:
                    assign_node.add_read_access(read_access)


class SparseFusionTransformer(ast.NodeTransformer):
    def __init__(self, directives):
        super().__init__()
        self.directives = directives

    def visit_FunctionDef(self, node):
        # rename function
        node.name = "executer"
        # add schedule argument
        node.args.args.append(ast.arg(arg="schedule"))

        # separate the loops into two groups: directive-driven
        preamble = []
        loop_bodies = {}

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):  # keep initializations
                preamble.append(stmt)
            elif isinstance(stmt, ast.For):
                # capture loops whose target is in directives
                loop_var = stmt.target.id
                if loop_var in self.directives:
                    loop_bodies[loop_var] = stmt
                    self.visit(stmt)
            else:
                # keep return for later
                ret_stmt = stmt

        # Build new structure
        new_for = ast.For(
            target=ast.Name(id="s", ctx=ast.Store()),
            iter=ast.Name(id="schedule", ctx=ast.Load()),
            body=[
                ast.For(
                    target=ast.Tuple(
                        elts=[ast.Name(id="first", ctx=ast.Store()),
                              ast.Name(id="second", ctx=ast.Store())],
                        ctx=ast.Store()
                    ),
                    iter=ast.Name(id="s", ctx=ast.Load()),
                    body=[
                        loop_bodies[self.directives[0]],
                        loop_bodies[self.directives[1]],
                    ],
                    orelse=[]
                )
            ],
            orelse=[]
        )

        node.body = preamble + [new_for, ret_stmt]
        return node
    def visit_For(self, node):
        if node.target.id == self.directives[0]:
            node.iter = ast.Name('first', ctx=ast.Load())
        if node.target.id == self.directives[1]:
            node.iter = ast.Name('second', ctx=ast.Load())


# class SparseFusionTransformer(ast.NodeTransformer):


class TileFusionLoopBodyExtractor(ast.NodeVisitor):
    def __init__(self, transformation):
        self.first_loop = None
        self.second_loop = None
        self.transformation = transformation
        self.ret_stmnt = None

    # TODO: Fix generic visits(use return value to have one return statement.)
    def visit_For(self, node):
        iter_var = self.visit(node.target)
        if iter_var == self.transformation[0]:
            self.first_loop = node.body
        elif iter_var == self.transformation[1]:
            self.second_loop = node.body
        self.generic_visit(node)

    def visit_Name(self, node):
        self.generic_visit(node)
        return node.id

    def visit_Return(self, node):
        print("TEST")
        self.ret_stmnt = node
        self.generic_visit(node)

class LegalFusionTransformer(ast.NodeTransformer):
    def __init__(self, transformation):
        self.target_loop = None
        self.target_loop_iter_var = None
        self.removed_iter_vars = []
        self.transformation = transformation

    # TODO: Fix generic visits(use return value to have one return statement.)
    def visit_For(self, node):
        iter_var = self.visit(node.target)
        if iter_var.id == self.transformation[0]:
            self.target_loop = node
            self.generic_visit(node)
            return node
        elif iter_var.id == self.transformation[1]:
            assert node.target is not None, "Wrong transformation."
            self.removed_iter_vars.append(iter_var.id)
            self.generic_visit(node)
            for child in node.body:
                self.target_loop.body.append(child)
            return None
        else:
            self.generic_visit(node)
            return node

    def visit_Name(self, node):
        if self.target_loop and node.id in self.removed_iter_vars:
            return ast.Name(id=self.transformation[0], ctx=node.ctx)
        else:
            return node


class SPFBuilder(ast.NodeVisitor, _SPFBuilderBase):

    def __init__(self):
        ast.NodeVisitor.__init__(self)
        _SPFBuilderBase.__init__(self)

    def visit_Module(self, node):
        assert len(node.body) == 1 and isinstance(node.body[0], ast.FunctionDef), "Wrong usage."
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        root = SPFNode("root")
        for s in node.body:
            assert isinstance(s, ast.For) or isinstance(s, ast.Assign) or isinstance(s, ast.Return), \
                "Only for loops and assignments are supported for now."
            node = self.visit(s)
            if not isinstance(s, ast.Return):
                root.add_child(node)
        return root

    def visit_Name(self, node):
        return node.id

    def visit_Constant(self, node):
        return str(node.value)

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Tuple assignment is not supported."
        target = self.convert_target(node.targets[0])
        value = self.visit(node.value)
        return self.create_assign_node(target, value)

    def visit_AugAssign(self, node):
        target = self.convert_target(node.target)
        value = self.visit(node.value)
        return self.create_assign_node(target, value, is_augmented=True)

    def visit_For(self, node):
        iter_var = self.visit(node.target)
        loop_bound = self.visit_iter(node.iter)
        body = []
        for s in node.body:
            assert (isinstance(s, ast.For) or isinstance(s, ast.Assign)
                    or isinstance(s, ast.AugAssign) or isinstance(s, ast.Return)) \
                , "Only for loops and assignments are supported for now."
            body.append(self.visit(s))
        return self.create_loop_node(iter_var, loop_bound, body)

    # TODO: Iter can be subscript two
    def visit_iter(self, node):
        if isinstance(node, ast.Call):
            if node.func.id == 'range' or node.func.id == 'prange':
                return self.visit_range(node)
            else:
                raise NotImplementedError(f"Unsupported function call: {node.func.id}")
        else:
            raise NotImplementedError(f"Unsupported iteration type: {type(node)}")

    def visit_range(self, node):
        assert len(node.args) <= 2, "Step is not supported in SPF yet."
        if len(node.args) == 1:
            start = ast.Constant(0)
            end = node.args[0]
        else:
            start = node.args[0]
            end = node.args[1]
        # assert ((isinstance(start, ast.Constant) or isinstance(start, ast.Name)) and
        #         (isinstance(end, ast.Constant) or isinstance(end, ast.Name))) or \
        #        (isinstance(start, ast.Subscript) and isinstance(end, ast.Subscript)), \
        #     "This functionality is not supported yet."

        start_sym = self.visit_loop_bound(start)
        end_sym = self.visit_loop_bound(end)
        loop_bound = LoopBound(start_sym, end_sym)
        return loop_bound

    def visit_loop_bound(self, node):
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            # Handle subscripted access to arrays or lists
            return self.visit_Subscript(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        else:
            raise NotImplementedError(f"Unsupported loop bound type: {type(node)}")

    # TODO: Figure out a way to handle array accesses
    def visit_Subscript(self, node):
        var = self.visit(node.value)
        index = self.visit(node.slice)
        return Access(var, index)

    def convert_target(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return self.visit(node)
        raise NotImplementedError(f"Unsupported assign target type: {type(node)}")

    def convert_write_target(self, target):
        if isinstance(target, str):
            return Access(target, None)
        if isinstance(target, Access):
            return target
        raise NotImplementedError(f"Unsupported write target type: {type(target)}")

    def visit_BinOp(self, node):
        expr = []
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.symbolize_operator(node.op)
        if isinstance(left, Expression):
            expr += left.elements
        else:
            expr.append(left)
        expr.append(op)
        if isinstance(right, Expression):
            expr += right.elements
        else:
            expr.append(right)
        return Expression(expr)

    def symbolize_operator(self, fn):
        operator_to_string = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%',
        }
        if type(fn) not in operator_to_string:
            raise NotImplementedError(f"Operator {fn} is not supported.")
        return operator_to_string[type(fn)]


class LCDToSPFConverter(_SPFBuilderBase):
    """Converts the LCD IR into the SPF loop/statement tree."""

    def __init__(self):
        super().__init__()

    def convert(self, lcd_ir):
        return self.visit(lcd_ir)

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, None)
        if visitor is None:
            raise NotImplementedError(f"Unsupported LCD node type: {type(node)}")
        return visitor(node)

    def visit_LCD(self, node):
        root = SPFNode("root")
        body, _, _ = self.convert_statement_list(node.body, [])
        for stmt in body:
            root.add_child(stmt)
        return root

    def visit_LCDFor(self, node):
        loop_bound = LoopBound(
            self.convert_value(node.lbound),
            self.convert_value(node.ubound),
        )
        body, _, break_triggers = self.convert_statement_list(node.body, [])
        break_constraints = [
            self.create_prior_iteration_break_premise(node, trigger)
            for trigger in break_triggers
        ]
        for child in body:
            self.add_domain_to_assignments(child, break_constraints)
        return self.create_loop_node(node.var.name, loop_bound, body)

    def visit_LCDAssign(self, node):
        return self.convert_assign(node, [])

    def convert_assign(self, node, domain):
        target = self.convert_target(node.target)
        value = self.convert_value(node.value)
        is_augmented = isinstance(node.value, LCDBin) and self._same_storage(node.target, node.value.left)
        return self.create_assign_node(
            target,
            value,
            is_augmented=is_augmented,
            domain=domain,
        )

    def visit_LCDIf(self, node):
        body, _, _ = self.convert_if(node, [])
        return body

    def convert_statement_list(self, statements, domain):
        body = []
        active_domain = list(domain)
        break_triggers = []
        for stmt in statements:
            if isinstance(stmt, LCDIf):
                if_body, active_domain, if_break_triggers = self.convert_if(stmt, active_domain)
                body.extend(if_body)
                break_triggers.extend(if_break_triggers)
            elif isinstance(stmt, LCDAssign):
                body.append(self.convert_assign(stmt, active_domain))
            elif isinstance(stmt, LCDFor):
                loop_node = self.visit_LCDFor(stmt)
                self.add_domain_to_assignments(loop_node, active_domain)
                body.append(loop_node)
            else:
                raise NotImplementedError(f"Unsupported LCD statement type: {type(stmt)}")
        return body, active_domain, break_triggers

    def convert_if(self, node, domain):
        condition = self.convert_condition(node.condition)
        body_domain = domain + [condition]
        body, _, break_triggers = self.convert_statement_list(node.if_body, body_domain)

        trigger = AndPremise(body_domain)
        next_domain = list(domain)
        if node.has_break:
            break_triggers.append(trigger)
            next_domain.append(NotPremise(trigger))
        if node.has_continue:
            next_domain.append(NotPremise(trigger))
        return body, next_domain, break_triggers

    def create_prior_iteration_break_premise(self, loop_node, trigger):
        iter_var = loop_node.var.name
        prior_iter_var = iter_var + "_prev"
        lower_bound = self.convert_value(loop_node.lbound)
        prior_bounds = [
            GreaterThan(prior_iter_var, lower_bound),
            GreaterThan(iter_var, prior_iter_var),
        ]
        prior_trigger = SubstitutePremise(trigger.copy(), {iter_var: prior_iter_var})
        return ForAllPremise(prior_iter_var, prior_bounds, NotPremise(prior_trigger))

    def add_domain_to_assignments(self, node, domain):
        if isinstance(node, AssignNode):
            self.add_domain_premises(node, domain)
            return
        for child in node.children:
            self.add_domain_to_assignments(child, domain)

    def convert_target(self, node):
        if isinstance(node, LCDVar):
            return node.name
        if isinstance(node, LCDAccess):
            return self.convert_access(node)
        raise NotImplementedError(f"Unsupported LCD assign target type: {type(node)}")

    def convert_write_target(self, target):
        if isinstance(target, str):
            return Access(target, None)
        if isinstance(target, Access):
            return target
        raise NotImplementedError(f"Unsupported LCD write target type: {type(target)}")

    def convert_condition(self, node):
        if isinstance(node, LCDCompare):
            left = self.convert_value(node.left)
            right = self.convert_value(node.right)
            comparison_to_premise = {
                "==": Equality,
                "!=": NotEqual,
                ">": GreaterThan,
                ">=": GreaterThanOrEqual,
                "<": LessThan,
                "<=": LessThanOrEqual,
            }
            if node.op not in comparison_to_premise:
                raise NotImplementedError(f"Unsupported LCD comparison operator: {node.op}")
            return comparison_to_premise[node.op](left, right)
        if isinstance(node, LCDBin) and node.op == "and":
            return AndPremise([self.convert_condition(node.left), self.convert_condition(node.right)])
        if isinstance(node, LCDBin) and node.op == "or":
            return OrPremise([self.convert_condition(node.left), self.convert_condition(node.right)])
        if isinstance(node, LCDUn) and node.op == "not":
            return NotPremise(self.convert_condition(node.target))
        return ExpressionPremise(self.convert_value(node))

    def convert_value(self, node):
        if isinstance(node, LCDVar):
            return node.name
        if isinstance(node, LCDConst):
            return str(node.value)
        if isinstance(node, LCDAccess):
            return self.convert_access(node)
        if isinstance(node, LCDBin):
            return self.convert_binary(node)
        if isinstance(node, LCDUn):
            target = self.convert_value(node.target)
            return Expression([node.op, target])
        if isinstance(node, LCDCompare):
            left = self.convert_value(node.left)
            right = self.convert_value(node.right)
            return Expression([left, node.op, right])
        raise NotImplementedError(f"Unsupported LCD expression type: {type(node)}")

    def convert_access(self, node):
        access = node.target.name
        for func in node.funcs:
            access = Access(access, self.convert_value(func))
        return access

    def convert_binary(self, node):
        expr = []
        left = self.convert_value(node.left)
        right = self.convert_value(node.right)
        if isinstance(left, Expression):
            expr += left.elements
        else:
            expr.append(left)
        expr.append(node.op)
        if isinstance(right, Expression):
            expr += right.elements
        else:
            expr.append(right)
        return Expression(expr)

    def _same_storage(self, left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, LCDVar):
            return left.name == right.name
        if isinstance(left, LCDAccess):
            return str(left) == str(right)
        return False


# DFS Algorithm on DFS tree
def extract_node_data(spf_node):
    # schedule_to_node = {}
    node_data = {}
    node_data[spf_node.id] = spf_node
    # Hold level, and statment_counter in addition to node to construct the schedule for each node correctly
    stack = [(spf_node, 0, -1, [])]
    schedule = []
    # loop_last_statements = {}
    while len(stack) > 0:
        cur_node, node_counter, cur_level, loop_domain = stack.pop()
        schedule = schedule[:cur_level * 2]
        level = cur_level + 1
        child_loop_domain = loop_domain
        if isinstance(cur_node, LoopNode):
            schedule.append(str(node_counter))
            cur_node.schedule = Schedule(schedule)
            node_data[cur_node.id] = cur_node
            schedule.append(cur_node.iter_var)
            child_loop_domain = loop_domain + cur_node.bound.to_premises(cur_node.iter_var)
        if isinstance(cur_node, AssignNode):
            schedule.append(str(node_counter))
            cur_node.schedule = Schedule(schedule)
            cur_node.loop_domain = loop_domain
            node_data[cur_node.id] = cur_node
        for i, c in enumerate(cur_node.children[::-1]):
            counter = len(cur_node.children) - 1 - i
            stack.append((c, counter, level, child_loop_domain))
    return node_data



test_code = """
n = len(A_indptr) - 1
for i in range(n):
    row_start = A_indptr[i]
    row_end = A_indptr[i + 1]
    for k_ptr in range(row_start, row_end):
        k = A_indices[k_ptr]
        if k >= i:
            break
        diag_val = 0.0
        for diag_ptr in range(A_indptr[k], A_indptr[k + 1]):
            if A_indices[diag_ptr] == k:
                diag_val = A_data[diag_ptr]
                break
        if diag_val != 0.0:
            A_data[k_ptr] = A_data[k_ptr] / diag_val
            l_ik = A_data[k_ptr]
            for j_ptr in range(k_ptr + 1, row_end):
                j = A_indices[j_ptr]
                for kj_ptr in range(A_indptr[k], A_indptr[k + 1]):
                    if A_indices[kj_ptr] == j:
                        u_kj = A_data[kj_ptr]
                        A_data[j_ptr] = A_data[j_ptr] - l_ik * u_kj
                        break
"""
