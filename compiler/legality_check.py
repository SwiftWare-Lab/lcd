import sympy

from copy import deepcopy
from sympy import Symbol, Eq, GreaterThan, And, Or, Not, StrictGreaterThan
from spf_raising import *
from SMT_Generator import generate_smt_file
from z3 import Solver, CheckSatResult, Z3_L_FALSE



def check_bound_equivalency(b1, b2):
    if isinstance(b1, Expression) and isinstance(b2, Expression):
        if len(b1.elements) != len(b2.elements):
            return False
        res = True
        for i, el in enumerate(b1.elements):
            res = res and check_bound_equivalency(el, b2.elements[i])
        return res
    if isinstance(b1, Access) and isinstance(b2, Access):
        if b1.var == b2.var:
            return check_bound_equivalency(b1.index, b2.index)
    elif not isinstance(b1, Access) and not isinstance(b2, Access):
        if b1 == b2:
            return True
        # elif (b1, b2) in mappings:
        #     return True
        else:
            return False
    else:
        return False


#TODO: For now it's fine to have frange on inner loops but this need to be fixed later.
def check_loop_bounds_safety(targeted_iters: tuple[str, str], loop_bounds: dict[str, LoopBound]):
    i1 = targeted_iters[0]
    i2 = targeted_iters[1]
    i1_start = loop_bounds[i1].start
    i1_end = loop_bounds[i1].end
    i2_start = loop_bounds[i2].start
    i2_end = loop_bounds[i2].end
    return check_bound_equivalency(i1_start, i2_start) and check_bound_equivalency(i1_end, i2_end)



#TODO: Fix it with new dependence premise design.
def extract_fusion_constraint_for_dependence(dependence: DataDependence):
    from_sched = dependence.from_si
    to_sched = dependence.to_si
    print(from_sched)
    print(to_sched)
    conditions = sympy.false
    base_cond = sympy.true
    for i, s in enumerate(from_sched.schedule):
         if s.isnumeric():
             assert(to_sched.schedule[i].isnumeric())
             if int(to_sched.schedule[i]) > int(from_sched.schedule[i]):
                 return conditions
             elif int(to_sched.schedule[i]) < int(from_sched.schedule[i]): #TODO: Think how I can fix this.
                     return True
         else:
             if not dependence.has_premise(Equality(from_sched.schedule[i], to_sched.schedule[i]))\
                     or dependence.has_premise(GreaterThanOrEqual(to_sched.schedule[i], from_sched.schedule[i])):
                 from_sym = Symbol(from_sched.schedule[i].replace(".", ""))
                 to_sym = Symbol(to_sched.schedule[i].replace(".", ""))
                 conditions = Or(And(base_cond, StrictGreaterThan(from_sym, to_sym)), conditions)
                 base_cond = And(Eq(from_sym, to_sym), base_cond)
    return conditions


def transform_sched(target_sched: Schedule, current_sched: Schedule, i1: str, i2: str):
    new_sched = deepcopy(current_sched)
    tar_index = 0
    print(i1, i2)
    for i, s in enumerate(target_sched.schedule):
        if s == i1:
            tar_index = i+1
            assert(new_sched.schedule[i] == i2)
            break
        if s.isnumeric():
            assert(new_sched.schedule[i].isnumeric())
            new_sched.schedule[i] = s
    new_sched.schedule[tar_index] = str(int(target_sched.schedule[tar_index]) + 1)
    return new_sched


# for now, we considered tuple holds the schedule order
# TODO: I should check that all the dependences after the transformation still hold(remove relevent dependences part
def extract_fusion_illegality_constraint(targeted_iters: tuple[str, str], dependences: list[DataDependence]):
    i1 = targeted_iters[0]
    i2 = targeted_iters[1]
    constraints = []
    for dep in dependences:
        # assert dep.to_si.occurs_in(i2) and dep.from_si.occurs_in(i1), "Dependences should be defined between given loops only"
        new_sched = transform_sched(dep.from_si, dep.to_si, i1, i2)
        new_dep = DataDependence(dep.from_si, new_sched, dep.data, dep.premises, dep.type)
        fusion_constraint_for_dependence = extract_fusion_constraint_for_dependence(new_dep)
        dependence_constraint = And(dep.get_condition_statement_sympy(),
                                    fusion_constraint_for_dependence)
        constraints.append(dependence_constraint)
    return constraints


# TODO: For now only works for legal fusion.
def check_legality_of_fusion(mapping: tuple[str, str],
                             node_data: dict[str, LoopNode]):
    loop_bounds = {}
    mapping_nodes = {}
    for lid, lnode in node_data.items():
        if isinstance(lnode, LoopNode):
            loop_bounds[lnode.iter_var] = lnode.bound
            if lnode.iter_var in mapping:
                mapping_nodes[lnode.iter_var] = lnode
    if check_loop_bounds_safety(mapping, loop_bounds):
        dependences = extract_dependences(node_data, mapping, mapping_nodes)
        # new_mapping = subscript_fused_vars(mapping)
        constraints = extract_fusion_illegality_constraint(mapping, dependences)
        return constraints, dependences
    else:
        return sympy.true
    # if check_loop_bounds_safety(targeted_iters, loop_bounds, mappings):
    #     return extract_fusion_illigality_constraint(targeted_iters, dependence, schedules), mappings
    # else:
    #     return sympy.true, []



# EXTRACTING DEPENDENCES

def  extract_latest_write_access_in_loop(loop_node):
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

# Works based on iter var to node data map.
def extract_raw_data_dependencies(node_data, last_write_accesses):
    schedule_to_reads = {}
    schedule_to_write = {}
    dependences = []
    for cur_node in node_data.values():
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
                    premises = [Equality(w_index[i], r_index[i]) for i in range(len(r_index))]
                    # if wi != ri:
                    raw_dependence = DataDependence(wi, ri, r_arr, premises, 'RAW')
                    dependences.append(raw_dependence)
    return dependences


def refine_dependences(dependences, fused_iter_vars):
    refined_dependences = []
    for dep in dependences:
        new_dep = update_dep(dep, fused_iter_vars)
        # subscript_dependence(new_dep)
        dep_cond = check_if_dependence_holds(new_dep)
        if dep_cond != sympy.false:
            refined_dependences.append(new_dep)
            dep.add_premises(dep_cond)
    return refined_dependences


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
    conditions = []
    for i, s in enumerate(from_sched.schedule):
        if s.isnumeric():
            assert (to_sched.schedule[i].isnumeric())
            if int(to_sched.schedule[i]) > int(from_sched.schedule[i]):
                return conditions
            elif int(to_sched.schedule[i]) < int(from_sched.schedule[i]):
                return conditions
        else:
            if not dependence.has_premise(Equality(from_sched.schedule[i], to_sched.schedule[i])):
                from_sym = from_sched.schedule[i].replace(".", "")
                to_sym = to_sched.schedule[i].replace(".", "")
                conditions.append(GreaterThanOrEqual(to_sym, from_sym))
    return conditions

def subscript_fused_vars(fused_vars):
    return fused_vars[0] + "1", fused_vars[1] + "2"

#TODO: For index subscriptions, we need to check if type of index is access and act accordingly(something like visit?)
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
        elif isinstance(fi, Access):
            new_fi = Access(fi.var, fi.index + "1")
            new_from_ind.append(new_fi)
        else:
            new_from_ind.append(fi)
    for j, s in enumerate(to_sched.schedule):
        if not s.isnumeric():
            new_to_sched.schedule[j] = s + "2"
    for k, ti in enumerate(dependence.to_ind):
        if ti in to_sched.schedule:
            new_to_ind.append(ti + "2")
        elif isinstance(ti, Access):
            new_ti = Access(ti.var, ti.index + "1")
            new_to_ind.append(new_ti)
        else:
            new_to_ind.append(ti)
    dependence.from_si = new_from_sched
    dependence.to_si = new_to_sched
    dependence.from_ind = tuple(new_from_ind)
    dependence.to_ind = tuple(new_to_ind)


# Works based on iter var to node data map.
def extract_dependences(node_data, fused_iter_vars, mapping_nodes):
    last_write_accesses = {}
    for loop_id in fused_iter_vars:
        last_write_accesses[loop_id] = []
        last_write_access = extract_latest_write_access_in_loop(mapping_nodes[loop_id])
        for s_node in last_write_access.values():
            last_write_accesses[loop_id].append(s_node)
    dependences = extract_raw_data_dependencies(node_data, last_write_accesses)
    # print(dependences)
    # fused_iter_vars = [node_data[loop_id].iter_var for loop_id in fused_loop_ids]
    refined_dependences = refine_dependences(dependences, fused_iter_vars)
    for i, dependence in enumerate(refined_dependences):
        print(dependence)
        print("-----------------")
    return refined_dependences

def finalize_legality_of_fusion(transformation, cur_ast, function_props):
    spf_builder = SPFBuilder()
    root_node = spf_builder.visit(cur_ast)
    node_data = extract_node_data(root_node)

    iv_node_data = {}
    for nid, node in node_data.items():
        if isinstance(node, LoopNode):
            iv_node_data[node.iter_var] = node
    constraints, dependencies = check_legality_of_fusion(transformation, node_data)
    if constraints[0] != sympy.false:
        smt_out = generate_smt_file(iv_node_data.values(), constraints[0], function_props)
        s = Solver()
        s.from_string(smt_out)
        if s.check() == CheckSatResult(Z3_L_FALSE):
           return True
        else:
            return dependencies
    else:
        return True


def transform_ast(transformation, cur_ast):
    fusion_transformer = LegalFusionTransformer(transformation)
    return fusion_transformer.visit(cur_ast)

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

src_sddmm_spmm = """
for i1 in frange(m):
    for i2 in range(Ap[i1], Ap[i1 + 1]):
        for i3 in range(k):
            Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
for j1 in frange(m):
    for j2 in range(Ap[j1], Ap[j1 + 1]):
        for j3 in range(n):
            Y[j1][j3] += Xx[j2] * D[j1][j3]
"""

src_sddmm_spmm_v2 = """
for i1 in range(m):
    for i2 in frange(Ap[i1], Ap[i1 + 1]):
        for i3 in range(k):
            Xx[i2] += B[i1][i3] * C[Ai[i2]][i3]
    for j2 in frange(Ap[i1], Ap[i1 + 1]):
        for j3 in range(n):
            Y[i1][j3] += Xx[j2] * D[i1][j3]
"""

# transformation: (i, j)
# dependence: DataDependence

# TODO: what about nested loops?
# Complete this.
def extract_dag_creation_components(transformation: tuple[str, str], dependence: DataDependence, node_data):
    iter_list = []
    iter_list.append(transformation[0])
    iter_list.append(transformation[1])
    from_ind = [_.left for _ in dependence.premises]
    to_ind = [_.right for _ in dependence.premises]
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


def generate_for_loop(iterator_var, start, end, tab_num):
    tab_str = "\t" * tab_num
    return "for {} in range({}, {}):\n{}".format(iterator_var, start, end, tab_str)

def generate_iteration_space(ordered_iter_list, node_data):
    code_str = ""
    tab_num = 2
    for iter_var in ordered_iter_list:
        for lid, lnode in node_data.items():
            if isinstance(lnode, LoopNode):
                if lnode.iter_var == iter_var:
                    code_str += generate_for_loop(iter_var, lnode.bound.start, lnode.bound.end, tab_num)
                    tab_num += 1
    return code_str


def extract_arguments(components, node_data):
    arguments = set()
    iter_list = components[0]
    symbolic_nodes = components[1]
    for it in iter_list:
        for lid, lnode in node_data.items():
            if isinstance(lnode, LoopNode):
                if lnode.iter_var == it:
                    bound_start = lnode.bound.start
                    bound_end = lnode.bound.end
                    if isinstance(bound_start, Access):
                        arguments.add(bound_start.get_data_array())
                    elif not bound_start.isdigit():
                        arguments.add(bound_start)
                    if isinstance(bound_end, Access):
                        arguments.add(bound_end.get_data_array())
                    elif not bound_end.isdigit():
                        arguments.add(bound_end)
    for sn in symbolic_nodes:
        if isinstance(sn, Access):
            arguments.add(sn.get_data_array())
    return arguments


def generate_dag_generation_code(arguments, components, node_data, partition_node_num):
    print(arguments)
    code_str = ""
    function_def = "def extract_dag({}):\n\t".format(', '. join(arguments))
    code_str += function_def
    symbolic_nodes = components[1]
    DAG_def = "DAG = [[] for i in range({})]\n\t".format(partition_node_num[symbolic_nodes[1]])
    code_str += DAG_def
    iteration_space_code = generate_iteration_space(components[0], node_data)
    code_str += iteration_space_code
    DAG_update = "DAG[{}].append({})\n".format(symbolic_nodes[1], symbolic_nodes[0])
    code_str += DAG_update
    return_stmnt = "\treturn DAG\n"
    code_str += return_stmnt
    return code_str



src = """
for i in range(10):
    for j in range(0,n):
        for k in range(Ap[i], Ap[i + 1]):
            A[i] = B[j] + C[k]
        
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
def spmm_spmm(m, n, Ap, Ai, Ax, B):
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
            for i3 in range(n):
                y[i1][i3] += Ax[i2] * B[Ai[i2]][i3]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
            for j3 in range(n):
                z[j1][j3] += Ax[j2] * y[Ai[j2]][j3]
"""

src_spmv_spmv = """
def spmv_spmv(m, n, Ap, Ai, Ax, B):
    for i1 in range(m):
        for i2 in range(Ap[i1], Ap[i1 + 1]):
                y[i1] += Ax[i2] * B[Ai[i2]]
    for j1 in range(m):
        for j2 in range(Ap[j1], Ap[j1 + 1]):
                z[j1] += Ax[j2] * y[Ai[j2]]
"""

src_gemm_spmm = """
def gemm_spmm(m, n, Ap, Ai, Ax, B, A):
    for i1 in range(m):
        for i2 in range(n):
            for i3 in range(n):
                y[i1][i3] += A[i1][i2] * B[i2][i3]
    for j1 in range(m):
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

src_sptrsv = """
for i in frange(n):
    x[i] = b[i]
    for j in range(IA[i], IA[i + 1]):
        x[i] -= A[j] * x[JA[j]]
    x[i] /= A[IA[i + 1] - 1]
"""

if __name__ == "__main__":
    tree = ast.parse(src_spmv_spmv)
    spf_builder = SPFBuilder()
    root_node = spf_builder.visit(tree)
    node_data = extract_node_data(root_node)
    #
    #     print(node_data)
    function_props = {
        'Ap': ['monotonically_increasing'],
        'Ai': [],
    }
    const, dependencies = check_legality_of_fusion(('i1', 'j1'), node_data)
    #     mappings = extract_mappings(root_node, node_data, spf_builder.fused_loop_ids)
    #     print(mappings)
    #     iv_node_data = {}
    #     for nid, node in node_data:
    #         if isinstance(node, LoopNode):
    #             iv_node_data[node.iter_var] = node
    #     print(check_legality_of_fusion(("i1", "j1"), iv_node_data))
    #     check_legality_of_fusion(spf_builder.fused_loop_ids, )
    #     fusion_transformer = LegalFusionTransformer()
    #     transformed_tree = fusion_transformer.visit(tree)
    #     print(ast.unparse(transformed_tree))
    print(dependencies)
    iter_list = extract_dag_creation_components(('i1', 'j1'), dependencies[0], node_data)
    print(iter_list)
    code = generate_dag_generation_code(iter_list, node_data, {'j1': 'm'})
    print(code)

    # new_file_name = "dag_extraction.py"
    #
    # with open(new_file_name, 'w') as py_file:
    #     py_file.write(code)
    #
    # import dag_extraction


    # print(ast.dump(ast.parse('l[1][3]', mode='eval'), indent=4))

# if __name__ == "__main__":
#     tree = ast.parse(src_sddmm_spmm_v2)
#     spf_builder = SPFBuilder()
#     root_node = spf_builder.visit(tree)
#     node_data = extract_node_data(root_node)
#
#     # print(node_data)
#     # dependencies = extract_dependences(node_data, spf_builder.fused_loop_ids)
#     # mappings = extract_mappings(root_node, node_data, spf_builder.fused_loop_ids)
#     # print(mappings)
#     iv_node_data = {}
#     for nid, node in node_data.items():
#         if isinstance(node, LoopNode):
#             iv_node_data[node.iter_var] = node
#     constraints = check_legality_of_fusion(("i2", "j2"), node_data)
#     if constraints[0] != sympy.false:
#     #For each constraint one separate smt file is generated.
#         smt_out = generate_smt_file(iv_node_data.values(), constraints[0], function_props)
#         s = Solver()
#         s.from_string(smt_out)
#         print(s.check())
#         if s.check() == CheckSatResult(Z3_L_FALSE):
#             fusion_transformer = LegalFusionTransformer()
#             transformed_tree = fusion_transformer.visit(tree)
#             print(ast.unparse(transformed_tree))
#     else:
#         fusion_transformer = LegalFusionTransformer()
#         transformed_tree = fusion_transformer.visit(tree)
#         print(ast.unparse(transformed_tree))