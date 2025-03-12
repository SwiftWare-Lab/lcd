import sympy
from sympy import And, Eq, Symbol, GreaterThan, LessThan, Function, Gt
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