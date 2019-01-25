from params import *
from math import sin,cos,pi
"""

X = ⎧ X1 ⟶ position of the cart
    ⎪ X2 ⟶ angle of pendulum
    ⎪ X3 ⟶ velocity of the cart
    ⎩ X4 ⟶ angular velocity of pendulum


Ẋ = ⎧ Ẋ1 = F1(X,U)
    ⎪ Ẋ2 = F2(X,U)
    ⎪ Ẋ3 = F3(X,U)
    ⎩ Ẋ4 = F4(X,U)

"""
def F1(X,U):
    return(X[2])
def F2(X,U):
    return(X[3])
def F3(X,U):
    return(
        (
            m2*L*sin(X[1])*(X[3]**2)
            - b1*X[2]
            - m2*gr*sin(2*X[1])/2
            + b2*cos(X[1])*X[3]/L
            + U
        )
        /
        (m1 + m2*(sin(X[1])**2))
    )
def F4(X,U):
    return(
        (
            -m2*sin(2*X[1])*(X[3]**2)/2
            + b1*cos(X[1])*X[2]/L
            + (m1+m2)*gr*sin(X[1])/L
            - (m1+m2)*b2*X[3]/(m2*(L**2))
            - cos(X[1])*U/L
        )
        /
        (m1 + m2*(sin(X[1])**2))
    )
