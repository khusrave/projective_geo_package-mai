from sympy import *
x, y, z = symbols('x y z')
init_printing(use_unicode=True)
f = diff(cos(x), x)
print(f)