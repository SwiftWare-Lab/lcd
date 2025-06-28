import sys

from cfg import CFGBuilder




# let's have an example code
src= """
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib_gen = fib()
for _ in range(10):
    next(fib_gen)
"""

src2= """
a = [0] * 10
for i in range(10):
    a[i] = 2 * a[i-2] + a[i]
"""

code_str = """
b = 10
if b>0:
    a = a+b
else:
    a = 10
print(a)
"""

cfg = CFGBuilder().build_from_src("Fib", src)
fun_cfg = cfg.functioncfgs.items()



cfg.build_visual('png')

# let's visualize a function cfg
fib_cfg = None
for (block_id, fun_name), fun_cfg in cfg.functioncfgs.items():
    if fun_name == "fib":
        fib_cfg = fun_cfg
g = fib_cfg.build_visual('png')
# show the graph
# if g:
#     g.view()


from SSA.const import SSA
from core.mnode import MNode

mnode = MNode("local") # create file name for given source code
mnode.source = code_str
mnode.gen_ast() # build abstract syntax tree
cfg = mnode.gen_cfg()

g = cfg.build_visual('png')
# show the graph
if g:
    g.view()

m_ssa = SSA()
ssa_results, const_dict = m_ssa.compute_SSA(cfg)

for block_id, stmt_res in ssa_results.items():
    print("These are the results for block ", block_id)
    print(stmt_res)

for name, value in const_dict.items():
    print(name, value)

print("SSA results:", ssa_results)