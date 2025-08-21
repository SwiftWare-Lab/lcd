import ast
from spf_def import *


# from sympy import Symbol, Eq, GreaterThan, Lt, And, Or, Not
# from copy import deepcopy
# from legality_check import check_legality_of_fusion

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


class SPFBuilder(ast.NodeVisitor):

    def __init__(self):
        # self.spf_tree = []
        # self.statements = []
        # self.schedules = [-1]
        # self.statements = []
        # self.current_schedule = []
        # self.loop_bounds = {}  # Maps loop variable names to their bounds
        self.loop_counter = 0
        self.stmt_counter = 0
        self.fused_loop_ids = []

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
        target = node.targets[0]
        stmt_id = "S" + str(self.stmt_counter)
        self.stmt_counter += 1
        assign_node = AssignNode(stmt_id, target, node.value)
        if isinstance(target, ast.Name):
            assign_node.write_access = Access(target.id, None)
        elif isinstance(target, ast.Subscript):
            assign_node.write_access = self.visit(target)
        else:
            raise NotImplementedError(f"Unsupported assign target type: {type(target)}")
        value = self.visit(node.value)
        if isinstance(value, Expression):
            for el in value.elements:
                if isinstance(el, Access):
                    assign_node.add_read_access(el)
        elif isinstance(value, Access):
            assign_node.add_read_access(value)
        return assign_node

    def visit_AugAssign(self, node):
        target = node.target
        stmt_id = "S" + str(self.stmt_counter)
        self.stmt_counter += 1
        assign_node = AssignNode(stmt_id, target, node.value)
        if isinstance(target, ast.Name):
            assign_node.write_access = Access(target.id, None)
        elif isinstance(target, ast.Subscript):
            assign_node.write_access = self.visit(target)
        else:
            raise NotImplementedError(f"Unsupported assign target type: {type(target)}")
        value = self.visit(node.value)
        if isinstance(value, Expression):
            for el in value.elements:
                if isinstance(el, Access):
                    assign_node.add_read_access(el)
        elif isinstance(value, Access):
            assign_node.add_read_access(value)
        return assign_node

    def visit_For(self, node):
        iter_var = self.visit(node.target)
        loop_bound = self.visit_iter(node.iter)
        loop_id = "F" + str(self.loop_counter)
        # if isinstance(node.iter, ast.Call) and node.iter.func.id == 'frange':
        #     self.fused_loop_ids.append(loop_id)
        self.loop_counter += 1
        loop_node = LoopNode(loop_id, iter_var, loop_bound)
        for s in node.body:
            assert (isinstance(s, ast.For) or isinstance(s, ast.Assign)
                    or isinstance(s, ast.AugAssign) or isinstance(s, ast.Return)) \
                , "Only for loops and assignments are supported for now."
            node = self.visit(s)
            loop_node.add_child(node)
        return loop_node

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
        assert ((isinstance(start, ast.Constant) or isinstance(start, ast.Name)) and
                (isinstance(end, ast.Constant) or isinstance(end, ast.Name))) or \
               (isinstance(start, ast.Subscript) and isinstance(end, ast.Subscript)), \
            "This functionality is not supported yet."

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
        else:
            raise NotImplementedError(f"Unsupported loop bound type: {type(node)}")

    # TODO: Figure out a way to handle array accesses
    def visit_Subscript(self, node):
        var = self.visit(node.value)
        index = self.visit(node.slice)
        return Access(var, index)

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


# DFS Algorithm on DFS tree
def extract_node_data(spf_node):
    # schedule_to_node = {}
    node_data = {}
    node_data[spf_node.id] = spf_node
    # Hold level, and statment_counter in addition to node to construct the schedule for each node correctly
    stack = [(spf_node, 0, -1)]
    schedule = []
    # loop_last_statements = {}
    while len(stack) > 0:
        cur_node, node_counter, cur_level = stack.pop()
        schedule = schedule[:cur_level * 2]
        level = cur_level + 1
        if isinstance(cur_node, LoopNode):
            schedule.append(str(node_counter))
            cur_node.schedule = Schedule(schedule)
            node_data[cur_node.id] = cur_node
            schedule.append(cur_node.iter_var)
        if isinstance(cur_node, AssignNode):
            schedule.append(str(node_counter))
            cur_node.schedule = Schedule(schedule)
            node_data[cur_node.id] = cur_node
        for i, c in enumerate(cur_node.children[::-1]):
            counter = len(cur_node.children) - 1 - i
            stack.append((c, counter, level))
    return node_data


