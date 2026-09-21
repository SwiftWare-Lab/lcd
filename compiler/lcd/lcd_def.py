class LCD:
    def __init__(self):
        self.body = [] #Statements
    
    def __str__(self):
        return '\n'.join(str(stmt) for stmt in self.body)

class LCDStatement:
    pass


"""
if condition{
    if_body
    break | continue | None
}
"""
class LCDIf(LCDStatement):
    def __init__(
            self,
            condition,
            if_body,
            has_break,
            has_continue,
            else_body=None,
            else_has_break=False,
            else_has_continue=False,
    ):
        self.condition = condition # CompareExpression
        self.if_body = if_body
        self.else_body = else_body or []
        self.has_break =  has_break
        self.has_continue = has_continue
        self.else_has_break = else_has_break
        self.else_has_continue = else_has_continue
    
    def __str__(self):
        result = f"if ({self.condition}){{\n"
        for stmt in self.if_body:
            # Indent each line of multi-line statements
            stmt_str = str(stmt)
            for line in stmt_str.split('\n'):
                result += f"    {line}\n"
        if self.has_break:
            result += "    break\n"
        elif self.has_continue:
            result += "    continue\n"
        result += "}"
        if self.else_body or self.else_has_break or self.else_has_continue:
            result += " else{\n"
            for stmt in self.else_body:
                stmt_str = str(stmt)
                for line in stmt_str.split('\n'):
                    result += f"    {line}\n"
            if self.else_has_break:
                result += "    break\n"
            elif self.else_has_continue:
                result += "    continue\n"
            result += "}"
        return result

"""
for var: bound_start -> bound_end{
    for_body
}
"""
class LCDFor(LCDStatement):
    def __init__(self, var, lbound, ubound, body):
        self.var = var
        self.lbound = lbound
        self.ubound = ubound
        self.body = body
    
    def __str__(self):
        result = f"for {self.var.name}: {self.lbound} -> {self.ubound}{{\n"
        for stmt in self.body:
            # Indent each line of multi-line statements
            stmt_str = str(stmt)
            for line in stmt_str.split('\n'):
                result += f"    {line}\n"
        result += "}"
        return result

"""
target = value
"""

class LCDAssign(LCDStatement):
    def __init__(self, target, value, source_statement=None):
        super().__init__()
        self.target = target # ValExpr
        self.value = value
        self.source_statement = source_statement
    
    def __str__(self):
        return f"{self.target} = {self.value}"


class LCDExpression:
    pass

class LCDVal(LCDExpression):
    pass

class LCDVar(LCDVal):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def __str__(self):
        return self.name

class LCDAccess(LCDVal):
    def __init__(self, target, *funcs):
        super().__init__()
        self.target = target # LCDVar
        self.funcs = funcs # list of LCDVal
    
    def __str__(self):
        indices = ''.join(f"[{func}]" for func in self.funcs)
        return f"{self.target}{indices}"
class LCDOp(LCDExpression):
    def __init__(self, op):
        super().__init__()
        self.op = op

class LCDCompare(LCDOp):
    def __init__(self, op, left, right):
        super().__init__(op)
        self.left = left # LCDExpression
        self.right = right # LCDExpression
    
    def __str__(self):
        return f"{self.left} {self.op} {self.right}"

class LCDBin(LCDOp):
    def __init__(self, op, left, right):
        super().__init__(op)
        self.left = left # LCDExpression
        self.right = right # LCDExpression
    
    def __str__(self):
        # Add parentheses for clarity in complex expressions
        left_str = f"({self.left})" if isinstance(self.left, (LCDBin, LCDCompare)) else str(self.left)
        right_str = f"({self.right})" if isinstance(self.right, (LCDBin, LCDCompare)) else str(self.right)
        return f"{left_str} {self.op} {right_str}"

class LCDUn(LCDOp):
    def __init__(self, op, target):
        super().__init__(op)
        self.target = target
    
    def __str__(self):
        # For 'not', use 'not expr', for others use 'op(expr)'
        if self.op == 'not':
            target_str = f"({self.target})" if isinstance(self.target, (LCDBin, LCDCompare)) else str(self.target)
            return f"{self.op} {target_str}"
        else:
            return f"{self.op}({self.target})"

class LCDConst(LCDExpression):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __str__(self):
        return str(self.value)


class LCDOpaque(LCDExpression):
    """Expression kept as text when the SPF backend only needs a placeholder."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def __str__(self):
        return self.text
