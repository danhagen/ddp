import numpy as np
from dynamics import *

def return_Phi(X,U,dt):
    """
    Takes in the state vector (X), the input vector (U) and returns the discretized and linearized state matrix, Phi.

    NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Create tests that ensure that X and U are the correct dimensions.
    [ ] - Create tests to make sure that the outputs are of the correct sizes.
    """
    assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
            and np.shape(X)==(4,)), "Error with the type and shape of X ["+ fnState_And_Control_Transition_Matrices.__name__+"()]."
    assert str(type(U)) in ["<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.float'>",
            "<class 'numpy.float64'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>"],\
        "U must be a number. Not " + str(type(U)) + "."

    # Removed the U split into two scalars because U is already a scalar.
    h = .00000001
    h1 = np.array([h,0,0,0])
    h2 = np.array([0,h,0,0])
    h3 = np.array([0,0,h,0])
    h4 = np.array([0,0,0,h])

    # Build the dFx matrix

    dFx = np.zeros((4,4))

    # dFx[0,0] = 0 # dF1/dx1⋅dx1 = (F1(X,U)-F1(X-h1,U))/h = 0
    # dFx[0,1] = 0 # dF1/dx2⋅dx2 = (F1(X,U)-F1(X-h2,U))/h = 0
    dFx[0,2] = 1 # dF1/dx3⋅dx3 = (F1(X,U)-F1(X-h3,U))/h = 1
    # dFx[0,3] = 0 # dF1/dx4⋅dx4 = (F1(X,U)-F1(X-h4,U))/h = 0

    # dFx[1,0] = 0 # dF2/dx1⋅dx2 = (F2(X,U)-F2(X-h1,U))/h = 0
    # dFx[1,1] = 0 # dF2/dx2⋅dx2 = (F2(X,U)-F2(X-h2,U))/h = 0
    # dFx[1,2] = 0 # dF2/dx3⋅dx2 = (F2(X,U)-F2(X-h3,U))/h = 1
    dFx[1,3] = 1 # dF2/dx4⋅dx2 = (F2(X,U)-F2(X-h4,U))/h = 0

    # F3 is the acceleration of the cart.
    dFx[2,0] = (F3(X,U)-F3(X-h1,U))/h
    dFx[2,1] = (F3(X,U)-F3(X-h2,U))/h
    dFx[2,2] = (F3(X,U)-F3(X-h3,U))/h
    dFx[2,3] = (F3(X,U)-F3(X-h4,U))/h

    # F4 is the angular acceleration of the pendulum.
    dFx[3,0] = (F4(X,U)-F4(X-h1,U))/h
    dFx[3,1] = (F4(X,U)-F4(X-h2,U))/h
    dFx[3,2] = (F4(X,U)-F4(X-h3,U))/h
    dFx[3,3] = (F4(X,U)-F4(X-h4,U))/h

    Phi = np.matrix(np.eye(4) + dFx*dt)
    assert np.shape(Phi)==(4,4) \
        and str(type(Phi))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
    "Phi must be a (4,4) numpy matrix. Not " + str(type(Phi)) + " of shape " + str(np.shape(Phi)) + "."
    return(Phi)

def return_B(X,U,dt):
    """
    Takes in the state vector (X), the input vector (U) and returns the discretized and linearized input matrix, B.

    NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Create tests that ensure that X and U are the correct dimensions.
    [ ] - Create tests to make sure that the outputs are of the correct sizes.
    """
    assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
            and np.shape(X)==(4,)), "Error with the type and shape of X ["+ fnState_And_Control_Transition_Matrices.__name__+"()]."
    assert str(type(U)) in ["<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.float'>",
            "<class 'numpy.float64'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>"],\
        "U must be a number. Not " + str(type(U)) + "."

    # Removed the U split into two scalars because U is already a scalar.
    h = .00000001
    h1 = np.array([h,0,0,0])
    h2 = np.array([0,h,0,0])
    h3 = np.array([0,0,h,0])
    h4 = np.array([0,0,0,h])

    #Build the dFu matrix

    dFu = np.zeros((4,1))

    dFu[0,0] = 0
    dFu[1,0] = 0
    dFu[2,0] = (F3(X,U)-F3(X,U-h))/h
    dFu[3,0] = (F4(X,U)-F4(X,U-h))/h

    B = np.matrix(dFu*dt)
    assert np.shape(B)==(4,1) \
            and str(type(B))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
        "B must be a (4,1) numpy matrix. Not " + str(type(B)) + " of shape " + str(np.shape(B)) + "."

    return(B)

def return_linearized_dynamics_matrices(X,U):
    """
    Takes in the state vector (X), the input vector (U) and outputs the linearized state and control matrices (dFx and dFu, respectively).

    NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Create tests that ensure that X and U are the correct dimensions.
    [ ] - Create tests to make sure that the outputs are of the correct sizes.
    """
    assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
            and np.shape(X)==(4,)), "Error with the type and shape of X ["+ fnState_And_Control_Transition_Matrices.__name__+"()]."

    # Removed the U split into two scalars because U is already a scalar.
    h = .00000001
    h1 = np.array([h,0,0,0])
    h2 = np.array([0,h,0,0])
    h3 = np.array([0,0,h,0])
    h4 = np.array([0,0,0,h])

    # Build the dFx matrix

    dFx = np.zeros((4,4))

    # dFx[0,0] = 0 # dF1/dx1⋅dx1
    # dFx[0,1] = 0 # dF1/dx2⋅dx2
    dFx[0,2] = 1 # dF1/dx3⋅dx3
    # dFx[0,3] = 0 # dF1/dx4⋅dx4

    # dFx[1,0] = 0 # dF2/dx1⋅dx2
    # dFx[1,1] = 0 # dF2/dx2⋅dx2
    # dFx[1,2] = 0 # dF2/dx3⋅dx2
    dFx[1,3] = 1 # dF2/dx4⋅dx2

    # F3 is the acceleration of the cart.
    dFx[2,0] = (F3(X,U)-F3(X-h1,U))/h
    dFx[2,1] = (F3(X,U)-F3(X-h2,U))/h
    dFx[2,2] = (F3(X,U)-F3(X-h3,U))/h
    dFx[2,3] = (F3(X,U)-F3(X-h4,U))/h

    # F4 is the angular acceleration of the pendulum.
    dFx[3,0] = (F4(X,U)-F4(X-h1,U))/h
    dFx[3,1] = (F4(X,U)-F4(X-h2,U))/h
    dFx[3,2] = (F4(X,U)-F4(X-h3,U))/h
    dFx[3,3] = (F4(X,U)-F4(X-h4,U))/h

    #Build the dFu matrix

    dFu = np.zeros((4,1))

    dFu[0,0] = 0
    dFu[1,0] = 0
    dFu[2,0] = (F3(X,U)-F3(X,U-h))/h
    dFu[3,0] = (F4(X,U)-F4(X,U-h))/h

    return(dFx,dFu)

def return_f32(X,U):
    return(
        (
            (
                m2*L*np.cos(X[1])*(X[3]**2)
                - m2*gr*np.cos(2*X[1])
                - b2*np.sin(X[1])*X[3]/L
            )
            *
            (
                m1 + m2*(np.sin(X[1])**2)
            )
            -
            (
                m2*L*np.sin(X[1])*(X[3]**2)
                - m2*gr*np.sin(2*X[1])/2
                - b1*X[2]
                + U
                + b2*np.cos(X[1])*X[3]/L
            )
            *
            (
                m2*np.sin(2*X[1])
            )
        )
        /
        (
            (
                m1 + m2*(np.sin(X[1])**2)
            )**2
        )
    )
def return_f33(X,U):
    return(
        (
            -b1
        )
        /
        (
            m1 + m2*(np.sin(X[1])**2)
        )
    )
def return_f34(X,U):
    return(
        (
            2*m2*L*np.sin(X[1])*X[3]
            + b2*np.cos(X[1])/L
        )
        /
        (
            m1 + m2*(np.sin(X[1])**2)
        )
    )

def return_f42(X,U):
    return(
        (
            (
                -m2*np.cos(2*X[1])*(X[3]**2)
                + (m1+m2)*gr*np.cos(X[1])/L
                - b1*np.sin(X[1])*X[2]/L
                + np.sin(X[1])*U/L
            )
            *
            (
                m1 + m2*(np.sin(X[1])**2)
            )
            -
            (
                -m2*np.sin(2*X[1])*(X[3]**2)/2
                + (m1+m2)*gr*np.sin(X[1])/L
                + b1*np.cos(X[1])*X[2]/L
                - (m1+m2)*b2*X[3]/(m2*(L**2))
                - np.cos(X[1])*U/L
            )
            *
            (
                m2*np.sin(2*X[1])
            )
        )
        /
        (
            (
                m1 + m2*(np.sin(X[1])**2)
            )**2
        )
    )
def return_f43(X,U):
    return(
        (
            b1*np.cos(X[1])/L
        )
        /
        (
            m1 + m2*(np.sin(X[1])**2)
        )
    )
def return_f44(X,U):
    return(
        (
            -m2*np.sin(2*X[1])*X[3]
            - (m1+m2)*b2/(m2*(L**2))
        )
        /
        (
            m1 + m2*(np.sin(X[1])**2)
        )
    )
