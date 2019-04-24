import numpy as np
import sympy as sp

gr = sp.Symbol('gr')

b1 = sp.Symbol('b1')

M = sp.Symbol('M')

L1 = sp.Symbol('L1')

R1 = sp.Symbol('R1')
R2 = sp.Symbol('R2')

x1 = sp.Symbol('X[0]')
x2 = sp.Symbol('X[1]')

u1 = sp.Symbol('U[0]')
u2 = sp.Symbol('U[1]')

x1_dot = x2
x2_dot = (
    -2*gr*sp.sin(x1)/L1
    - 4*b1*x2/(M*L1**2)
    + 4*R1*u1/(M*L1**2)
    + 4*R2*u2/(M*L1**2)
)

print(
    "def F1(X,U):\n"
    + "\treturn("
    + str(x1_dot)
    + ")"
)
print(
    "def F2(X,U):\n"
    + "\treturn("
    + str(x2_dot)
    + ")"
)
"""
OUTPUT:
def F1(X,U):
    return(X[1])
def F2(X,U):
    return(
        -2*gr*sin(X[0])/L1
        + 4*R1*U[0]/(L1**2*M)
        + 4*R2*U[1]/(L1**2*M)
        - 4*X[1]*b1/(L1**2*M)
    )
"""
