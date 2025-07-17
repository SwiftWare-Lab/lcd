import sympy

class DependencePremise:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __eq__(self, other):
        if type(other) == type(self):
            return other.left == self.left and other.right == self.right
        else:
            return False

    def get_sympy_expression(self):
        raise NotImplementedError("This functionality is only implemented for subclass objects.")

    def __repr__(self):
        raise NotImplementedError("This functionality is only implemented for subclass objects.")

    def __str__(self):
        return self.__repr__()

class Equality(DependencePremise):
    def get_sympy_expression(self):
        return sympy.Eq(sympy.Symbol(str(self.left)), sympy.Symbol(str(self.right)))

    def __repr__(self):
        return str(self.left) + ' = ' + str(self.right)

    def __eq__(self, other):
        if type(other) == type(self):
            return (other.left == self.left and other.right == self.right) and \
                (other.left == self.right and other.right == self.left)
        else:
            return False

class GreaterThanOrEqual(DependencePremise):
    def get_sympy_expression(self):
        return sympy.Gt(sympy.Symbol(str(self.left)), sympy.Symbol(str(self.right)))

    def __repr__(self):
        return str(self.left) + ' >= ' + str(self.right)

class GreaterThan(DependencePremise):
    def get_sympy_expression(self):
        return sympy.StrictGreaterThan(sympy.Symbol(str(self.left)), sympy.Symbol(str(self.right)))

    def __repr__(self):
        return str(self.left) + ' > ' + str(self.right)

class LoopBound:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def get_sympy_expr(self, iter_var):
        f1 = []
        f2 = []
        consts = []
        if isinstance(self.start, Access):
            f1, start = self.start.get_sympy_expr()
        elif self.start.isnumeric():
            start = int(self.start)
        else:
            start = sympy.Symbol(self.start)
            consts.append(start)
        if isinstance(self.end, Access):
            f2, end = self.end.get_sympy_expr()
        elif self.end.isnumeric():
            end = int(self.end)
        else:
            end = sympy.Symbol(self.end)
            consts.append(end)
        return consts, f1 + f2, sympy.And(sympy.Gt(sympy.Symbol(iter_var), start), sympy.LessThan(sympy.Symbol(iter_var), end))

    def __repr__(self):
        return '(' + str(self.start) + ', ' + str(self.end) + ')'


class Schedule:

    def __init__(self, schedule):
        self.schedule = schedule

    def occurs_in(self, iter_var: str) -> bool:
        return iter_var in self.schedule

    def __str__(self):
        return '(' + ', '.join(self.schedule) + ')'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __copy__(self):
        new_sched = [s for s in self.schedule]
        return Schedule(new_sched)


class Access:
    def __init__(self, var, index):
        self.var = var
        self.index = index

    def get_sympy_expr(self):
        func = self.get_data_array()
        funcs = [func]
        args_sympy = self.get_index_sympy_expr()
        args = args_sympy[1]
        funcs += [f for f in self.get_index_sympy_expr()[0] if f is not None]
        func_sym = sympy.Function(func)(args)
        return funcs, func_sym

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
        else:
            return self.var

    def get_index(self):
        if isinstance(self.var, Access):
            return self.var.get_index() + (self.index,)
        else:
            return (self.index,)

    def get_index_sympy_expr(self):
        if isinstance(self.index, Access):
            return self.index.get_sympy_expr()
        elif isinstance(self.index, Expression):
            return [], self.index.get_sympy_expr()
        else:
            return [], sympy.Symbol(self.index.replace(".", ""))


class Expression:
    def __init__(self, elements):
        self.elements = elements

    def __str__(self):
        return ' '.join(list(map(str, self.elements)))

    def __repr__(self):
        return str(self)

    def get_sympy_expr(self):
        return sympy.parse_expr(str(self).replace(".", "").replace('[', '(').replace(']', ')'))


class DataDependence:
    def __init__(self, from_si, to_si, data, initial_premises, type):
        self.from_si = from_si
        self.to_si = to_si
        self.data = data
        # self.from_ind = from_ind
        # self.to_ind = to_ind
        self.premises = initial_premises
        self.type = type
        # assert len(from_ind) == len(to_ind)

    def add_premises(self, premises):
        self.premises.extend(premises)

    def has_premise(self, condition):
        return condition in self.premises

    def get_condition_statement_sympy(self):
        condition = sympy.true
        for premise in self.premises:
            condition = sympy.And(premise.get_sympy_expression(), condition)
        return condition

    def __repr__(self):
        condition = ' and '.join(map(str, self.premises))
        return str(self.from_si) + '->' + str(
            self.to_si) + '\ndata: `' + self.data.__repr__() + '`\ncondition: ' + condition + '\ntype: ' + self.type


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
        # print(loop_bound)


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
