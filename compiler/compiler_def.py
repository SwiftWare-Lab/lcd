import sympy
from sympy import And, Eq, Symbol, GreaterThan, LessThan, Function, Gt


class LoopBound:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    # def get_sympy_expr(self, iter_var):
    #     f1 = []
    #     f2 = []
    #     consts = []
    #     if isinstance(self.start, Access):
    #         f1, start = self.start.get_sympy_expr()
    #     elif self.start.isnumeric():
    #         start = int(self.start)
    #     else:
    #         start = Symbol(self.start)
    #         consts.append(start)
    #     if isinstance(self.end, Access):
    #         f2, end = self.end.get_sympy_expr()
    #     elif self.end.isnumeric():
    #         end = int(self.end)
    #     else:
    #         end = Symbol(self.end)
    #         consts.append(end)
    #     return consts, f1 + f2, And(Gt(Symbol(iter_var), start), LessThan(Symbol(iter_var), end))

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
        func_sym = Function(func)(args)
        return funcs, func_sym

    def __eq__(self, other):
        # if type(self.index) is type(other.index) or type(self.var) is type(other.var):
        #     return False
        if type(self.index) is list and type(other.index) is list:
            self.index.sort()
            other.index.sort()
        return self.var == other.var and self.index == other.index

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
            return [], Symbol(self.index.replace(".", ""))


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
    def __init__(self, from_si, to_si, data, from_ind, to_ind, type):
        self.from_si = from_si
        self.to_si = to_si
        self.data = data
        self.from_ind = from_ind
        self.to_ind = to_ind
        self.type = type
        assert len(from_ind) == len(to_ind)

    def has_condition(self, i1, i2):
        for i in range(len(self.from_ind)):
            if not isinstance(self.from_ind[i], Access) and not isinstance(self.to_ind[i], Access):
                if (self.from_ind[i] == i1 and self.to_ind[i] == i2) or (self.from_ind[i] == i2 and self.to_ind[i] == i1):
                    return True
        return False

    def get_condition_statement_sympy(self):
        condition = sympy.true
        for i in range(len(self.from_ind)):
            from_sym = Symbol(str(self.from_ind[i]).replace(".", ""))
            to_sym = Symbol(str(self.to_ind[i]).replace(".", ""))
            condition = And(Eq(from_sym, to_sym), condition)
        return condition

    def __repr__(self):
        conditions = []
        for i in range(len(self.from_ind)):
            conditions.append('{} == {}'.format(self.from_ind[i], self.to_ind[i]))
        condition = ' and '.join(conditions)
        return str(self.from_si) + '->' + str(
            self.to_si) + '\ndata: `' + self.data.__repr__() + '`\ncondition: ' + condition + '\ntype: ' + self.type

