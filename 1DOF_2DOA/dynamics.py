from params import *
from physiology.muscle_params_BIC_TRI import *
from math import sin,cos,pi
"""
X = ⎧ X1 ⟶ angle of pendulum
    ⎩ X2 ⟶ angular velocity of pendulum

Ẋ = ⎧ Ẋ1 ⟶ F1(X,U)
    ⎩ Ẋ2 ⟶ F2(X,U)
"""
def F1(X,U):
    return(X[1])
def F2(X,U):
    return(
        -2*gr*sin(X[0])/L1
        + 4*R1(X)*U[0]/(L1**2*M)
        + 4*R2(X)*U[1]/(L1**2*M)
        - 4*X[1]*b1/(L1**2*M)
    )
