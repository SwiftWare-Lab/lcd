import ast
from spf_def import *


# from sympy import Symbol, Eq, GreaterThan, Lt, And, Or, Not
# from copy import deepcopy
# from legality_check import check_legality_of_fusion

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
        if isinstance(node.iter, ast.Call) and node.iter.func.id == 'frange':
            self.fused_loop_ids.append(loop_id)
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
            if node.func.id == 'range' or node.func.id == 'frange':
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


# transformation: (i, j)
# dependence: DataDependence

# TODO: what about nested loops?
# Complete this.
def generate_dag_creation_code(transformation: tuple[str, str], dependence: DataDependence, node_data):
    iter_list = []
    iter_list.append(transformation[0])
    iter_list.append(transformation[1])
    from_ind = dependence.from_ind
    to_ind = dependence.to_ind
    new_transformation = [transformation[0], transformation[1]]
    for i, ind in enumerate(from_ind):
        assert ind != transformation[1] and to_ind[i] != transformation[0], "Data Error!"
        if ind == transformation[0]:
            if isinstance(to_ind[i], Access):
                ufc = to_ind[i]
                iter_list.remove(ind)
                if ufc.index not in iter_list:
                    iter_list.append(ufc.index)
                new_transformation[0] = to_ind[i]
        if to_ind[i] == transformation[1]:
            if isinstance(ind, Access):
                ufc = ind
                iter_list.remove(to_ind[i])
                if ufc.index not in iter_list:
                    iter_list.append(ufc.index)
                new_transformation[1] = ind
    iter_var_to_node = {}
    for node_id, node in node_data.items():
        if node_id.startswith("F"):
            iter_var_to_node[node.iter_var] = node
    ordered_iter_list = []
    for iter_var in iter_list:
        node = iter_var_to_node[iter_var]
        lb = node.bound.start
        ub = node.bound.end
        new_iter_var1 = add_extra_iter_var_based_on_bound(lb, iter_list, iter_var_to_node)
        if new_iter_var1:
            iter_list.append(new_iter_var1)
            ordered_iter_list.append(new_iter_var1)
        new_iter_var2 = add_extra_iter_var_based_on_bound(ub, iter_list, iter_var_to_node)
        if new_iter_var2:
            iter_list.append(new_iter_var2)
            ordered_iter_list.append(new_iter_var2)
        if iter_var not in ordered_iter_list:
            ordered_iter_list.append(iter_var)
    return ordered_iter_list, new_transformation


def add_extra_iter_var_based_on_bound(bound, iter_list, iter_var_to_node):
    if isinstance(bound, Access):
        if isinstance(bound.index, Expression):
            for e in bound.index.elements:
                if e in iter_var_to_node and e not in iter_list:
                    return e
        elif bound.index in iter_var_to_node and bound.index not in iter_list:
            return bound.index
    return None


src = """
for i in range(10):
    for j in range(0,n):
        for k in range(Ap[i], Ap[i + 1]):
            A[i] = B[j] + C[k]
        
"""

src_sddmm_spmm = """
for i1 in range(m):
    for i2 in frange(Ap[i1], Ap[i1 + 1]):
        for i3 in range(k):
            Xx[i2] += B[i1 * k + i3] * C[Ai[i2]][i3]
for j1 in range(m):
    for j2 in frange(Ap[j1], Ap[j1 + 1]):
        for j3 in range(n):
            Y[j1][j3] += Xx[j2] * D[j1][j3]
"""

src_spmm_spmm_v2 = """
for i3 in frange(n):
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
for j1 in frange(m):
    for j2 in range(Ap[j1], Ap[j1 + 1]):
        for j3 in range(n):
            z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
"""

src_spmm_spmm = """
for i1 in frange(m):
    for i2 in range(Ap[i1], Ap[i1 + 1]):
        for i3 in range(n):
            y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
for j1 in frange(m):
    for j2 in range(Ap[j1], Ap[j1 + 1]):
        for j3 in range(n):
            z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
"""

src_spmm_spmm_v3 = """
for i1 in frange(m):
    for i2 in range(Ap[i1], Ap[i1 + 1]):
        for i3 in range(n):
            y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
for j3 in frange(n):
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
"""

src_spmm_spmm_v4 = """
for i3 in frange(n):
    for i2 in range(Ap[i1], Ap[i1 + 1]):
        for i1 in range(m):
            y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
for j3 in frange(n):
    for j2 in range(Ap[j1], Ap[j1 + 1]):
        for j1 in range(m):
            z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
"""

src_sptrsv = """
for i in frange(n):
    x[i] = b[i]
    for j in range(IA[i], IA[i + 1]):
        x[i] -= A[j] * x[JA[j]]
    x[i] /= A[IA[i + 1] - 1]
"""

# if __name__ == "__main__":
#     tree = ast.parse(src_sddmm_spmm)
#     spf_builder = SPFBuilder()
#     root_node = spf_builder.visit(tree)
#     node_data = extract_node_data(root_node)
#
#     print(node_data)
#     # dependencies = extract_dependences(node_data, spf_builder.fused_loop_ids)
#     mappings = extract_mappings(root_node, node_data, spf_builder.fused_loop_ids)
#     print(mappings)
#     iv_node_data = {}
#     for nid, node in node_data:
#         if isinstance(node, LoopNode):
#             iv_node_data[node.iter_var] = node
#     print(check_legality_of_fusion(("i1", "j1"), iv_node_data))
# check_legality_of_fusion(spf_builder.fused_loop_ids, )
# fusion_transformer = LegalFusionTransformer()
# transformed_tree = fusion_transformer.visit(tree)
# print(ast.unparse(transformed_tree))

# iter_list = generate_dag_creation_code(('i1', 'j1'), dependencies[0], node_data)
# print(iter_list)
# print(ast.dump(ast.parse('l[1][3]', mode='eval'), indent=4))
