import ast
from compiler_def import *
from sympy import Symbol, Eq, GreaterThan, Lt, And, Or, Not
from copy import deepcopy


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
        if isinstance(node.iter, ast.Call) and node.iter.func.id == 'frange':
            self.fused_loop_ids.append(loop_id)
        self.loop_counter += 1
        loop_node = LoopNode(loop_id, iter_var, loop_bound)
        for s in node.body:
            assert isinstance(s, ast.For) or isinstance(s, ast.Assign) or isinstance(s, ast.AugAssign), \
                "Only for loops and assignments are supported for now."
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


#DFS Algorithm on DFS tree
def extract_node_data(spf_node):
    # schedule_to_node = {}
    node_data = {}
    #Hold level, and statment_counter in addition to node to construct the schedule for each node correctly
    stack = [(spf_node, 0, -1)]
    schedule = []
    loop_last_statements = {}
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

def extract_latest_write_access_in_loop(loop_node):
    last_write_access = {}
    #BFS
    queue = [loop_node]
    while len(queue) > 0:
        cur_node = queue.pop(0)
        if isinstance(cur_node, AssignNode):
            last_write_access[cur_node.write_access] = cur_node
        for c in cur_node.children:
            queue.append(c)
    return last_write_access

def extract_raw_data_dependencies(node_data, last_write_accesses):
    schedule_to_reads = {}
    schedule_to_write = {}
    dependences = []
    for schedule, cur_node in node_data.items():
        if isinstance(cur_node, AssignNode):
            schedule_to_reads[cur_node.schedule] = cur_node.read_accesses
    for last_write_access in last_write_accesses.values():
        for s_node in last_write_access:
            schedule_to_write[s_node.schedule] = s_node.write_access
    for ri, r_accesses in schedule_to_reads.items():
        for wi, w_access in schedule_to_write.items():
            for r_access in r_accesses:
                r_arr = r_access.get_data_array()
                w_arr = w_access.get_data_array()
                r_index = r_access.get_index()
                w_index = w_access.get_index()
                if r_arr == w_arr:
                    # if wi != ri:
                    raw_dependence = DataDependence(wi, ri, r_arr, w_index, r_index, 'RAW')
                    dependences.append(raw_dependence)
    return dependences


def refine_dependences(dependences, fused_iter_vars):
    refined_dependences = []
    extra_constrains = []
    for dep in dependences:
        new_dep = update_dep(dep, fused_iter_vars)
        subscript_dependence(new_dep)
        dep_cond = check_if_dependence_holds(new_dep)
        if dep_cond != sympy.false:
            refined_dependences.append(new_dep)
            extra_constrains.append(dep_cond)
    return refined_dependences, extra_constrains


def update_dep(dep, fused_iter_vars):
    new_dep = deepcopy(dep)
    from_si = new_dep.from_si.schedule
    to_si = new_dep.to_si.schedule
    for i, fi in enumerate(from_si):
        if fi in fused_iter_vars:
            new_dep.from_si.schedule = from_si[:i+2]
            break
    for j, ti in enumerate(to_si):
        if ti in fused_iter_vars:
            new_dep.to_si.schedule = to_si[:j+2]
            break
    return new_dep


def check_if_dependence_holds(dependence):
    from_sched = dependence.from_si
    to_sched = dependence.to_si
    conditions = sympy.false
    base_cond = sympy.true
    for i, s in enumerate(from_sched.schedule):
        if s.isnumeric():
            assert (to_sched.schedule[i].isnumeric())
            if int(to_sched.schedule[i]) > int(from_sched.schedule[i]):
                return Or(conditions, base_cond)
            elif int(to_sched.schedule[i]) < int(from_sched.schedule[i]):  # TODO: Think how I can fix this.
                return conditions
        else:
            if not dependence.has_condition(from_sched.schedule[i], to_sched.schedule[i]):
                from_sym = Symbol(from_sched.schedule[i].replace(".", ""))
                to_sym = Symbol(to_sched.schedule[i].replace(".", ""))
                conditions = Or(And(base_cond, GreaterThan(to_sym, from_sym)), conditions)
                base_cond = And(Eq(from_sym, to_sym), base_cond)
    return conditions


def subscript_dependence(dependence):
    from_sched = dependence.from_si
    to_sched = dependence.to_si
    new_from_sched = deepcopy(from_sched)
    new_to_sched = deepcopy(to_sched)
    new_from_ind = []
    new_to_ind = []
    for i, s in enumerate(from_sched.schedule):
        if not s.isnumeric():
            new_from_sched.schedule[i] = s + "1"
    for k, fi in enumerate(dependence.from_ind):
        if fi in from_sched.schedule:
            new_from_ind.append(fi + "1")
        else:
            new_from_ind.append(fi)
    for j, s in enumerate(to_sched.schedule):
        if not s.isnumeric():
            new_to_sched.schedule[j] = s + "2"
    for k, ti in enumerate(dependence.to_ind):
        if ti in to_sched.schedule:
            new_to_ind.append(ti + "2")
        else:
            new_to_ind.append(ti)
    dependence.from_si = new_from_sched
    dependence.to_si = new_to_sched
    dependence.from_ind = tuple(new_from_ind)
    dependence.to_ind = tuple(new_to_ind)


def extract_dependences(node_data, fused_loop_ids):
    last_write_accesses = {}
    for loop_id in fused_loop_ids:
        last_write_accesses[loop_id] = []
        last_write_access = extract_latest_write_access_in_loop(node_data[loop_id])
        for s_node in last_write_access.values():
            last_write_accesses[loop_id].append(s_node)
    dependences = extract_raw_data_dependencies(node_data, last_write_accesses)
    fused_iter_vars = [node_data[loop_id].iter_var for loop_id in fused_loop_ids]
    refined_dependences, extra_constraints = refine_dependences(dependences, fused_iter_vars)
    for i, dependence in enumerate(refined_dependences):
            print(dependence)
            print(extra_constraints[i])
            print("-----------------")


src = """
for i in range(10):
    for j in range(0,n):
        for k in range(Ap[i], Ap[i + 1]):
            A[i] = B[j] + C[k]
        
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

src_sptrsv = """
for i in frange(n):
    x[i] = b[i]
    for j in range(IA[i], IA[i + 1]):
        x[i] -= A[j] * x[JA[j]]
    x[i] /= A[IA[i + 1] - 1]
"""


tree = ast.parse(src_sptrsv)
spf_builder = SPFBuilder()
node = spf_builder.visit(tree)
schedule_to_node = extract_node_data(node)
extract_dependences(schedule_to_node, spf_builder.fused_loop_ids)



# print(ast.dump(ast.parse('l[1][3]', mode='eval'), indent=4))