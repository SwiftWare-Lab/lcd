import ast
from compiler.lcd.lcd_def import *


class LCDBuilder(ast.NodeVisitor):
    """Converts Python AST to LCD AST"""

    def __init__(self):
        pass

    def visit_Module(self, node):
        assert len(node.body) == 1 and isinstance(node.body[0], ast.FunctionDef), "Wrong usage."
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        lcd = LCD()
        for s in node.body:
            if isinstance(s, ast.Return):
                continue
            lcd_stmt = self.visit(s)
            if lcd_stmt is not None:
                lcd.body.append(lcd_stmt)
        return lcd

    def visit_Name(self, node):
        return LCDVar(node.id)

    def visit_Constant(self, node):
        return LCDConst(node.value)

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Tuple assignment is not supported."
        target = self.visit_expression(node.targets[0])
        value = self.visit_expression(node.value)
        return LCDAssign(target, value)

    def visit_AugAssign(self, node):
        # Convert AugAssign to regular assignment with binary operation
        target = self.visit_expression(node.target)
        value = self.visit_expression(node.value)
        op = self.symbolize_operator(node.op)
        # Create binary operation: target = target op value
        bin_op = LCDBin(op, target, value)
        return LCDAssign(target, bin_op)

    def visit_If(self, node):
        condition = self.visit_expression(node.test)
        if_body = []
        has_break = False
        has_continue = False
        
        for s in node.body:
            if isinstance(s, ast.Break):
                has_break = True
                continue
            elif isinstance(s, ast.Continue):
                has_continue = True
                continue
            elif isinstance(s, ast.Return):
                continue
            stmt = self.visit(s)
            if stmt is not None:
                if_body.append(stmt)
        
        return LCDIf(condition, if_body, has_break, has_continue)

    def visit_Break(self, node):
        # Break is handled within If nodes
        return None

    def visit_Continue(self, node):
        # Continue is handled within If nodes
        return None

    def visit_For(self, node):
        iter_var = self.visit(node.target)
        if not isinstance(iter_var, LCDVar):
            raise NotImplementedError(f"Loop variable must be LCDVar, got {type(iter_var)}")
        lbound, ubound = self.visit_iter(node.iter)
        body = []
        for s in node.body:
            if isinstance(s, ast.Return):
                continue
            stmt = self.visit(s)
            if stmt is not None:
                body.append(stmt)
        return LCDFor(iter_var, lbound, ubound, body)

    def visit_iter(self, node):
        if isinstance(node, ast.Call):
            if node.func.id == 'range' or node.func.id == 'prange':
                return self.visit_range(node)
            else:
                raise NotImplementedError(f"Unsupported function call: {node.func.id}")
        else:
            raise NotImplementedError(f"Unsupported iteration type: {type(node)}")

    def visit_range(self, node):
        assert len(node.args) <= 2, "Step is not supported in LCD yet."
        if len(node.args) == 1:
            start = ast.Constant(0)
            end = node.args[0]
        else:
            start = node.args[0]
            end = node.args[1]
        lbound = self.visit_expression(start)
        ubound = self.visit_expression(end)
        # Return lbound and ubound separately
        return (lbound, ubound)

    def visit_Subscript(self, node):
        target = self.visit_expression(node.value)
        index = self.visit_expression(node.slice)
        
        # if not isinstance(index, LCDVal):
        #     raise NotImplementedError(f"Subscript index must be LCDVal, got {type(index)}")
        
        # Extract base variable and collect all indices
        if isinstance(target, LCDVar):
            # Simple case: a[i] -> LCDAccess(LCDVar('a'), i)
            base_var = target
            funcs = [index]
        elif isinstance(target, LCDAccess):
            # Nested case: a[i][j] -> LCDAccess(LCDVar('a'), i, j)
            # Extract the base variable from the nested access
            base_var = target.target
            if not isinstance(base_var, LCDVar):
                raise NotImplementedError(f"LCDAccess target must be LCDVar, got {type(base_var)}")
            # Collect all previous indices plus the new one
            funcs = list(target.funcs) + [index]
        else:
            raise NotImplementedError(f"Subscript target must be LCDVar or LCDAccess, got {type(target)}")
        
        # Create LCDAccess with base variable and all indices as funcs
        return LCDAccess(base_var, *funcs)

    def visit_BinOp(self, node):
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        op = self.symbolize_operator(node.op)
        return LCDBin(op, left, right)

    def visit_Compare(self, node):
        assert len(node.ops) == 1, "Multiple comparisons not supported yet."
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.comparators[0])
        op = self.symbolize_comparison_operator(node.ops[0])
        return LCDCompare(op, left, right)

    def visit_BoolOp(self, node):
        assert len(node.values) >= 2, "BoolOp must have at least 2 values."
        op = self.symbolize_bool_operator(node.op)
        left = self.visit_expression(node.values[0])
        right = self.visit_expression(node.values[1])
        result = LCDBin(op, left, right)
        # Handle multiple values (chain operations)
        for i in range(2, len(node.values)):
            right = self.visit_expression(node.values[i])
            result = LCDBin(op, result, right)
        return result

    def visit_UnaryOp(self, node):
        target = self.visit_expression(node.operand)
        op = self.symbolize_unary_operator(node.op)
        return LCDUn(op, target)

    def visit_expression(self, node):
        """Visit an expression node and return LCDExpression"""
        if isinstance(node, ast.Name):
            return self.visit_Name(node)
        elif isinstance(node, ast.Constant):
            return self.visit_Constant(node)
        elif isinstance(node, ast.Subscript):
            return self.visit_Subscript(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_BinOp(node)
        elif isinstance(node, ast.Compare):
            return self.visit_Compare(node)
        elif isinstance(node, ast.BoolOp):
            return self.visit_BoolOp(node)
        elif isinstance(node, ast.UnaryOp):
            return self.visit_UnaryOp(node)
        else:
            raise NotImplementedError(f"Unsupported expression type: {type(node)}")

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

    def symbolize_comparison_operator(self, op):
        operator_to_string = {
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>=',
            ast.Is: 'is',
            ast.IsNot: 'is not',
            ast.In: 'in',
            ast.NotIn: 'not in',
        }
        if type(op) not in operator_to_string:
            raise NotImplementedError(f"Comparison operator {op} is not supported.")
        return operator_to_string[type(op)]

    def symbolize_bool_operator(self, op):
        operator_to_string = {
            ast.And: 'and',
            ast.Or: 'or',
        }
        if type(op) not in operator_to_string:
            raise NotImplementedError(f"Boolean operator {op} is not supported.")
        return operator_to_string[type(op)]

    def symbolize_unary_operator(self, op):
        operator_to_string = {
            ast.Not: 'not',
            ast.UAdd: '+',
            ast.USub: '-',
        }
        if type(op) not in operator_to_string:
            raise NotImplementedError(f"Unary operator {op} is not supported.")
        return operator_to_string[type(op)]


if __name__ == "__main__":
    test_code = """
def ilu0_csr_numba(A_data, A_indices, A_indptr, n):
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
    
    # Parse the Python code into AST
    tree = ast.parse(test_code)
    
    # Convert to LCD AST
    builder = LCDBuilder()
    lcd_ast = builder.visit(tree)
    
    # Print the LCD IR
    print(lcd_ast)
