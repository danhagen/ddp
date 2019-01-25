import numpy as np
# from Function3 import *
# from Function4 import *
from dynamics import *
import sympy as sp

# global m2 m1 L g
x1,x2,x3,x4 = sp.Symbol('x1'),sp.Symbol('x2'),sp.Symbol('x3'),sp.Symbol('x4')
dx1,dx2,dx3,dx4 = sp.Symbol('dx1'),sp.Symbol('dx2'),sp.Symbol('dx3'),sp.Symbol('dx4')
# m2,m1,L,gr = sp.Symbol('m2'),sp.Symbol('m1'),sp.Symbol('L'),sp.Symbol('gr')
# F,A,B = sp.Symbol('F'),sp.Symbol('A'),sp.Symbol('B')

h = .00001
A = np.zeros((4,4),dtype='float64')

#Build the A matrix
#A[0,0] = 0 dF1/x1 dx1 = x3
#A[0,1] = 0 dF1/x2 dx1 = x3
A[0,2] = 1 #dF1/x3 dx1 = x3
#A[0,3] = 0 dF1/x4 dx1 = x3

#A[1,0] = 0 dF2/x1 dx2 = x4
#A[1,1] = 0 dF2/x2 dx2 = x4
#A[1,2] = 0 dF2/x3 dx2 = x4
A[1,3] = 1 #dF2/x4 dx2 = x4

#F3 is the acceleration of the cart.
A[2,0] = (Function3([x1,x2,x3,x4],F)-Function3([x1-h,x2,x3,x4],F))/h
A[2,1] = (Function3([x1,x2,x3,x4],F)-Function3([x1,x2-h,x3,x4],F))/h
A[2,2] = (Function3([x1,x2,x3,x4],F)-Function3([x1,x2,x3-h,x4],F))/h
A[2,3] = (Function3([x1,x2,x3,x4],F)-Function3([x1,x2,x3,x4-h],F))/h

#F4 is the angular acceleration of the pendulum.
A[3,0] = (Function4([x1,x2,x3,x4],F)-Function4([x1-h,x2,x3,x4],F))/h
A[3,1] = (Function4([x1,x2,x3,x4],F)-Function4([x1,x2-h,x3,x4],F))/h
A[3,2] = (Function4([x1,x2,x3,x4],F)-Function4([x1,x2,x3-h,x4],F))/h
A[3,3] = (Function4([x1,x2,x3,x4],F)-Function4([x1,x2,x3,x4-h],F))/h

#Build the B matrix
B = np.zeros((4,1),dtype='float64')
B[0,0] = 0
B[1,0] = 0
B[2,0] = (Function3([x1,x2,x3,x4],F)-Function3([x1,x2,x3,x4],F-h))/h
B[3,0] = (Function4([x1,x2,x3,x4],F)-Function4([x1,x2,x3,x4],F-h))/h
