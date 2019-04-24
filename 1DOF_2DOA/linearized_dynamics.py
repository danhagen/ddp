import numpy as np
from params import *
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
            and np.shape(X)==(2,)), "Error with the type and shape of X ["+ return_Phi.__name__+"()]."
    assert (str(type(U)) in ["<class 'numpy.ndarray'>"]
            and np.shape(U)==(2,)), "Error with the type and shape of U ["+ return_Phi.__name__+"()]."

    # Removed the U split into two scalars because U is already a scalar.

    h1 = np.array([h,0])
    h2 = np.array([0,h])

    # Build the dFx matrix

    dFx = np.zeros((2,2))

    # dFx[0,0] = 0 # dF1/dx1⋅dx1 = (F1(X,U)-F1(X-h1,U))/h = 0
    dFx[0,1] = 1 # dF1/dx4⋅dx4 = (F1(X,U)-F1(X-h2,U))/h = 1

    # F2 is the angular acceleration.
    dFx[1,0] = (F2(X,U)-F2(X-h1,U))/h
    dFx[1,1] = (F2(X,U)-F2(X-h2,U))/h

    Phi = np.matrix(np.eye(2) + dFx*dt)
    assert np.shape(Phi)==(2,2) \
        and str(type(Phi))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
    "Phi must be a (2,2) numpy matrix. Not " + str(type(Phi)) + " of shape " + str(np.shape(Phi)) + "."
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
            and np.shape(X)==(2,)), "Error with the type and shape of X ["+ return_B.__name__+"()]."
    assert (str(type(U)) in ["<class 'numpy.ndarray'>"]
            and np.shape(U)==(2,)), "Error with the type and shape of U ["+ return_Phi.__name__+"()]."

    # Removed the U split into two scalars because U is already a scalar.
    h1 = np.array([h,0])
    h2 = np.array([0,h])

    #Build the dFu matrix

    dFu = np.zeros((2,2))

    dFu[0,0] = 0
    dFu[0,1] = 0

    dFu[1,0] = (F2(X,U)-F2(X,U-h1))/h
    dFu[1,1] = (F2(X,U)-F2(X,U-h2))/h

    B = np.matrix(dFu*dt)
    assert np.shape(B)==(2,2) \
            and str(type(B))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
        "B must be a (2,2) numpy matrix. Not " + str(type(B)) + " of shape " + str(np.shape(B)) + "."

    return(B)
def return_linearized_dynamics_matrices(X,U,dt):
    """
    Takes in the input U and the the corresponding output X, as well as dt and returns two lists that contain the linearized dynamic matrices for each timestep for range(len(Time)-1).

    Note that if np.shape(X)[1] = N and len(U) = M, then N = M + 1 (i.e., there is one more timestep for output than input since the initial conditions are assigned to the first state space timestep). Therefore, we only concern ourselves with the linearized dynamics of the (N-1) steps where U drives X to the next timestep (i.e., X will only go up to the N-1 step or index X[:,:-1].)

    Phi is a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.

    B is a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.

    ### NEEDS TO BE TESTED ###

    np.shape(X)[1] == len(U)+1

    len(Phi) == len(U)
    type(Phi) == list
    len(B) == len(U)
    type(B) == list

    ##########################
    """
    Phi = list(
            map(
                lambda X,U: return_Phi(X,U,dt),
                X[:,:-1].T,
                U.T
            )
        )

    B = list(
            map(
                lambda X,U: return_B(X,U,dt),
                X[:,:-1].T,
                U.T
            )
        )
    return(Phi,B)
