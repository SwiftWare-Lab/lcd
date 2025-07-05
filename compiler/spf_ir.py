import ast
from compiler_def import *


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

    def visit_Module(self, node):
        root = SPFNode("root")
        for s in node.body:
            assert isinstance(s, ast.For) or isinstance(s, ast.Assign), \
                "Only for loops and assignments are supported for now."
            root.add_child(self.visit(s))
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
        self.loop_counter += 1
        loop_node = LoopNode(loop_id, iter_var, loop_bound)
        for s in node.body:
            assert isinstance(node, ast.For) or isinstance(node, ast.Assign), \
                "Only for loops and assignments are supported for now."
            node = self.visit(s)
            loop_node.add_child(node)
        return loop_node

    # TODO: Iter can be subscript two
    def visit_iter(self, node):
        if isinstance(node, ast.Call):
            if node.func.id == 'range':
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


class SPFNode:
    def __init__(self, node_id):
        self.id = node_id
        self.children = []

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
        self.loop_bound = loop_bound


class AssignNode(SPFNode):
    def __init__(self, node_id, target, value):
        super().__init__(node_id)
        self.target = target
        self.value = value
        self.read_accesses = []
        self.write_access = None

    def add_read_access(self, read_access):
        self.read_accesses.append(read_access)

    def __str__(self):
        return (self.id + "\n - write: " + str(self.write_access) +
                "\n - reads: \n" + '\n'.join(map(str, self.read_accesses)))


src = """
for i in range(10):
    for j in range(0,n):
        for k in range(Ap[i], Ap[i + 1]):
            A[i] = B[j] + C[k]
        
"""

src_spmm_spmm = """
for i1 in range(m):
    for i2 in range(Ap[i1], Ap[i1 + 1]):
        for i3 in range(n):
            y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
for j1 in range(m):
    for j2 in range(Ap[j1], Ap[j1 + 1]):
        for j3 in range(n):
            z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
"""

#DFS Algorithm on DFS tree
def traverse_tree(spf_node):
    schedule_to_node = {}
    #Hold level, and statment_counter in addition to node to construct the schedule for each node correctly
    stack = [(spf_node, 0, -1)]
    schedule = []
    while len(stack) > 0:
        cur_node, node_counter, cur_level = stack.pop()
        schedule = schedule[:cur_level * 2]
        level = cur_level + 1
        if isinstance(cur_node, LoopNode):
            schedule.append(str(node_counter))
            schedule_to_node[tuple(schedule)] = cur_node
            schedule.append(cur_node.iter_var)
        if isinstance(cur_node, AssignNode):
            schedule.append(str(node_counter))
            schedule_to_node[tuple(schedule)] = cur_node
        for i, c in enumerate(cur_node.children[::-1]):
            counter = len(cur_node.children) - 1 - i
            stack.append((c, counter, level))
    return schedule_to_node



tree = ast.parse(src_spmm_spmm)
spf_builder = SPFBuilder()
node = spf_builder.visit(tree)
schedule_to_node = traverse_tree(node)

for key, value in schedule_to_node.items():
    print(key, value)

# print(ast.dump(ast.parse('l[1][3]', mode='eval'), indent=4))