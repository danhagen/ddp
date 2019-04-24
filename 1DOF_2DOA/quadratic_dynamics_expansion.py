# import numpy as np
# from params import *
# from dynamics import *
# #### NEEDS TO BE CHANGED FOR DBL PENDULUM
# def return_Phi(X,U,dt):
#     """
#     Takes in the state vector (X), the input vector (U) and returns the discretized and linearized state matrix, Phi.
#
#     NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).
#
#     #######################
#     ##### NEED TO DO: #####
#     #######################
#
#     [ ] - Create tests that ensure that X and U are the correct dimensions.
#     [ ] - Create tests to make sure that the outputs are of the correct sizes.
#     """
#     assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
#             and np.shape(X)==(6,)), "Error with the type and shape of X ["+ fnState_And_Control_Transition_Matrices.__name__+"()]."
#     assert str(type(U)) in ["<class 'int'>",
#             "<class 'float'>",
#             "<class 'numpy.float'>",
#             "<class 'numpy.float64'>",
#             "<class 'numpy.int32'>",
#             "<class 'numpy.int64'>"],\
#         "U must be a number. Not " + str(type(U)) + "."
#
#     # Removed the U split into two scalars because U is already a scalar.
#
#     h1 = np.array([h,0,0,0])
#     h2 = np.array([0,h,0,0])
#     h3 = np.array([0,0,h,0])
#     h4 = np.array([0,0,0,h])
#
#     # Build the dFx matrix
#
#     dFx = np.zeros((4,4))
#
#     # dFx[0,0] = 0 # dF1/dx1⋅dx1 = (F1(X,U)-F1(X-h1,U))/h = 0
#     # dFx[0,1] = 0 # dF1/dx2⋅dx2 = (F1(X,U)-F1(X-h2,U))/h = 0
#     dFx[0,2] = 1 # dF1/dx3⋅dx3 = (F1(X,U)-F1(X-h3,U))/h = 1
#     # dFx[0,3] = 0 # dF1/dx4⋅dx4 = (F1(X,U)-F1(X-h4,U))/h = 0
#
#     # dFx[1,0] = 0 # dF2/dx1⋅dx2 = (F2(X,U)-F2(X-h1,U))/h = 0
#     # dFx[1,1] = 0 # dF2/dx2⋅dx2 = (F2(X,U)-F2(X-h2,U))/h = 0
#     # dFx[1,2] = 0 # dF2/dx3⋅dx2 = (F2(X,U)-F2(X-h3,U))/h = 1
#     dFx[1,3] = 1 # dF2/dx4⋅dx2 = (F2(X,U)-F2(X-h4,U))/h = 0
#
#     # F3 is the acceleration of the cart.
#     dFx[2,0] = (F3(X,U)-F3(X-h1,U))/h
#     dFx[2,1] = (F3(X,U)-F3(X-h2,U))/h
#     dFx[2,2] = (F3(X,U)-F3(X-h3,U))/h
#     dFx[2,3] = (F3(X,U)-F3(X-h4,U))/h
#
#     # F4 is the angular acceleration of the pendulum.
#     dFx[3,0] = (F4(X,U)-F4(X-h1,U))/h
#     dFx[3,1] = (F4(X,U)-F4(X-h2,U))/h
#     dFx[3,2] = (F4(X,U)-F4(X-h3,U))/h
#     dFx[3,3] = (F4(X,U)-F4(X-h4,U))/h
#
#     Phi = np.matrix(np.eye(4) + dFx*dt)
#     assert np.shape(Phi)==(4,4) \
#         and str(type(Phi))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
#     "Phi must be a (4,4) numpy matrix. Not " + str(type(Phi)) + " of shape " + str(np.shape(Phi)) + "."
#     return(Phi)
# def return_B(X,U,dt):
#     """
#     Takes in the state vector (X), the input vector (U) and returns the discretized and linearized input matrix, B.
#
#     NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).
#
#     #######################
#     ##### NEED TO DO: #####
#     #######################
#
#     [ ] - Create tests that ensure that X and U are the correct dimensions.
#     [ ] - Create tests to make sure that the outputs are of the correct sizes.
#     """
#     assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
#             and np.shape(X)==(4,)), "Error with the type and shape of X ["+ fnState_And_Control_Transition_Matrices.__name__+"()]."
#     assert str(type(U)) in ["<class 'int'>",
#             "<class 'float'>",
#             "<class 'numpy.float'>",
#             "<class 'numpy.float64'>",
#             "<class 'numpy.int32'>",
#             "<class 'numpy.int64'>"],\
#         "U must be a number. Not " + str(type(U)) + "."
#
#     # Removed the U split into two scalars because U is already a scalar.
#
#     h1 = np.array([h,0,0,0])
#     h2 = np.array([0,h,0,0])
#     h3 = np.array([0,0,h,0])
#     h4 = np.array([0,0,0,h])
#
#     #Build the dFu matrix
#
#     dFu = np.zeros((4,1))
#
#     dFu[0,0] = 0
#     dFu[1,0] = 0
#     dFu[2,0] = (F3(X,U)-F3(X,U-h))/h
#     dFu[3,0] = (F4(X,U)-F4(X,U-h))/h
#
#     B = np.matrix(dFu*dt)
#     assert np.shape(B)==(4,1) \
#             and str(type(B))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
#         "B must be a (4,1) numpy matrix. Not " + str(type(B)) + " of shape " + str(np.shape(B)) + "."
#
#     return(B)
# def return_Fxx(X,U,dt):
#     """
#     Takes in the state vector (X), the input vector (U) and returns the discretized and linearized state matrix, Phi.
#
#     NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).
#
#     #######################
#     ##### NEED TO DO: #####
#     #######################
#
#     [ ] - Create tests that ensure that X and U are the correct dimensions.
#     [ ] - Create tests to make sure that the outputs are of the correct sizes.
#     """
#     assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
#             and np.shape(X)==(4,)), "Error with the type and shape of X ["+ return_Fxx.__name__+"()]."
#     assert str(type(U)) in ["<class 'int'>",
#             "<class 'float'>",
#             "<class 'numpy.float'>",
#             "<class 'numpy.float64'>",
#             "<class 'numpy.int32'>",
#             "<class 'numpy.int64'>"],\
#         "U must be a number. Not " + str(type(U)) + "."
#
#     # Removed the U split into two scalars because U is already a scalar.
#
#     h1 = np.array([h,0,0,0,0,0])
#     h2 = np.array([0,h,0,0,0,0])
#     h3 = np.array([0,0,h,0,0,0])
#     h4 = np.array([0,0,0,h,0,0])
#     h5 = np.array([0,0,0,0,h,0])
#     h6 = np.array([0,0,0,0,0,h])
#
#     # Build the dFx matrix
#
#     f_xx = np.zeros(6**3).reshape(6,6,6)
#
#     # f_xx[0,:,:] = (return_Phi(X_o,U,dt)-return_Phi(X_o-h1,U,dt))/h
#     # f_xx[1,:,:] = (return_Phi(X_o,U,dt)-return_Phi(X_o-h2,U,dt))/h
#     # f_xx[2,:,:] = (return_Phi(X_o,U,dt)-return_Phi(X_o-h3,U,dt))/h
#     # f_xx[3,:,:] = (return_Phi(X_o,U,dt)-return_Phi(X_o-h4,U,dt))/h
#     # f_xx = np.swapaxes(f_xx,1,2)
#
#     # d2F1 = np.zeros((4,4))
#
#     # d2F2 = np.zeros((4,4))
#
#     # d2F3 is the jerk of the cart.
#     # f_xx[0,0,2] = ((F3(X,U)-F3(X-h1,U))-(F3((X-h1),U)-F3((X-h1)-h1,U)))/(h**2)
#     # f_xx[0,1,2] = ((F3(X,U)-F3(X-h1,U))-(F3((X-h2),U)-F3((X-h2)-h1,U)))/(h**2)
#     # f_xx[0,2,2] = ((F3(X,U)-F3(X-h1,U))-(F3((X-h3),U)-F3((X-h3)-h1,U)))/(h**2)
#     # f_xx[0,3,2] = ((F3(X,U)-F3(X-h1,U))-(F3((X-h4),U)-F3((X-h4)-h1,U)))/(h**2)
#     #
#     # f_xx[1,0,2] = ((F3(X,U)-F3(X-h2,U))-(F3((X-h1),U)-F3((X-h1)-h2,U)))/(h**2)
#     # f_xx[1,1,2] = ((F3(X,U)-F3(X-h2,U))-(F3((X-h2),U)-F3((X-h2)-h2,U)))/(h**2)
#     # f_xx[1,2,2] = ((F3(X,U)-F3(X-h2,U))-(F3((X-h3),U)-F3((X-h3)-h2,U)))/(h**2)
#     # f_xx[1,3,2] = ((F3(X,U)-F3(X-h2,U))-(F3((X-h4),U)-F3((X-h4)-h2,U)))/(h**2)
#     #
#     # f_xx[2,0,2] = ((F3(X,U)-F3(X-h3,U))-(F3((X-h1),U)-F3((X-h1)-h3,U)))/(h**2)
#     # f_xx[2,1,2] = ((F3(X,U)-F3(X-h3,U))-(F3((X-h2),U)-F3((X-h2)-h3,U)))/(h**2)
#     # f_xx[2,2,2] = ((F3(X,U)-F3(X-h3,U))-(F3((X-h3),U)-F3((X-h3)-h3,U)))/(h**2)
#     # f_xx[2,3,2] = ((F3(X,U)-F3(X-h3,U))-(F3((X-h4),U)-F3((X-h4)-h3,U)))/(h**2)
#     #
#     # f_xx[3,0,2] = ((F3(X,U)-F3(X-h4,U))-(F3((X-h1),U)-F3((X-h1)-h4,U)))/(h**2)
#     # f_xx[3,1,2] = ((F3(X,U)-F3(X-h4,U))-(F3((X-h2),U)-F3((X-h2)-h4,U)))/(h**2)
#     # f_xx[3,2,2] = ((F3(X,U)-F3(X-h4,U))-(F3((X-h3),U)-F3((X-h3)-h4,U)))/(h**2)
#     # f_xx[3,3,2] = ((F3(X,U)-F3(X-h4,U))-(F3((X-h4),U)-F3((X-h4)-h4,U)))/(h**2)
#
#     f_xx[0,0,2] = 0
#     f_xx[0,1,2] = 0
#     f_xx[0,2,2] = 0
#     f_xx[0,3,2] = 0
#
#     f_xx[1,0,2] = 0
#     f_xx[1,1,2] = return_f322(X,U)
#     f_xx[1,2,2] = return_f323(X,U)
#     f_xx[1,3,2] = return_f324(X,U)
#
#     f_xx[2,0,2] = 0
#     f_xx[2,1,2] = return_f332(X,U)
#     f_xx[2,2,2] = return_f333(X,U)
#     f_xx[2,3,2] = return_f334(X,U)
#
#     f_xx[3,0,2] = 0
#     f_xx[3,1,2] = return_f342(X,U)
#     f_xx[3,2,2] = return_f343(X,U)
#     f_xx[3,3,2] = return_f344(X,U)
#
#     # d2F4 is the angular jerk of the pendulum.
#     # f_xx[0,0,3] = ((F4(X,U)-F4(X-h1,U))-(F4((X-h1),U)-F4((X-h1)-h1,U)))/(h**2)
#     # f_xx[0,1,3] = ((F4(X,U)-F4(X-h1,U))-(F4((X-h2),U)-F4((X-h2)-h1,U)))/(h**2)
#     # f_xx[0,2,3] = ((F4(X,U)-F4(X-h1,U))-(F4((X-h3),U)-F4((X-h3)-h1,U)))/(h**2)
#     # f_xx[0,3,3] = ((F4(X,U)-F4(X-h1,U))-(F4((X-h4),U)-F4((X-h4)-h1,U)))/(h**2)
#     #
#     # f_xx[1,0,3] = ((F4(X,U)-F4(X-h2,U))-(F4((X-h1),U)-F4((X-h1)-h2,U)))/(h**2)
#     # f_xx[1,1,3] = ((F4(X,U)-F4(X-h2,U))-(F4((X-h2),U)-F4((X-h2)-h2,U)))/(h**2)
#     # f_xx[1,2,3] = ((F4(X,U)-F4(X-h2,U))-(F4((X-h3),U)-F4((X-h3)-h2,U)))/(h**2)
#     # f_xx[1,3,3] = ((F4(X,U)-F4(X-h2,U))-(F4((X-h4),U)-F4((X-h4)-h2,U)))/(h**2)
#     #
#     # f_xx[2,0,3] = ((F4(X,U)-F4(X-h3,U))-(F4((X-h1),U)-F4((X-h1)-h3,U)))/(h**2)
#     # f_xx[2,1,3] = ((F4(X,U)-F4(X-h3,U))-(F4((X-h2),U)-F4((X-h2)-h3,U)))/(h**2)
#     # f_xx[2,2,3] = ((F4(X,U)-F4(X-h3,U))-(F4((X-h3),U)-F4((X-h3)-h3,U)))/(h**2)
#     # f_xx[2,3,3] = ((F4(X,U)-F4(X-h3,U))-(F4((X-h4),U)-F4((X-h4)-h3,U)))/(h**2)
#     #
#     # f_xx[3,0,3] = ((F4(X,U)-F4(X-h4,U))-(F4((X-h1),U)-F4((X-h1)-h4,U)))/(h**2)
#     # f_xx[3,1,3] = ((F4(X,U)-F4(X-h4,U))-(F4((X-h2),U)-F4((X-h2)-h4,U)))/(h**2)
#     # f_xx[3,2,3] = ((F4(X,U)-F4(X-h4,U))-(F4((X-h3),U)-F4((X-h3)-h4,U)))/(h**2)
#     # f_xx[3,3,3] = ((F4(X,U)-F4(X-h4,U))-(F4((X-h4),U)-F4((X-h4)-h4,U)))/(h**2)
#     f_xx[0,0,3] = 0
#     f_xx[0,1,3] = 0
#     f_xx[0,2,3] = 0
#     f_xx[0,3,3] = 0
#
#     f_xx[1,0,3] = 0
#     f_xx[1,1,3] = return_f422(X,U)
#     f_xx[1,2,3] = return_f423(X,U)
#     f_xx[1,3,3] = return_f424(X,U)
#
#     f_xx[2,0,3] = 0
#     f_xx[2,1,3] = return_f432(X,U)
#     f_xx[2,2,3] = return_f433(X,U)
#     f_xx[2,3,3] = return_f434(X,U)
#
#     f_xx[3,0,3] = 0
#     f_xx[3,1,3] = return_f442(X,U)
#     f_xx[3,2,3] = return_f443(X,U)
#     f_xx[3,3,3] = return_f444(X,U)
#
#     Fxx = f_xx*dt
#     assert np.shape(Fxx)==(6,6,6) , "Fxx must be a (6,6,6) numpy array. Not shape " + str(np.shape(Fxx)) + "."
#     return(Fxx)
# def return_Fxu(X,U,dt):
#     """
#     Takes in the state vector (X), the input vector (U) and returns the discretized and linearized state matrix, Phi.
#
#     NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).
#
#     #######################
#     ##### NEED TO DO: #####
#     #######################
#
#     [ ] - Create tests that ensure that X and U are the correct dimensions.
#     [ ] - Create tests to make sure that the outputs are of the correct sizes.
#     """
#     assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
#             and np.shape(X)==(4,)), "Error with the type and shape of X ["+ return_Fxx.__name__+"()]."
#     assert str(type(U)) in ["<class 'int'>",
#             "<class 'float'>",
#             "<class 'numpy.float'>",
#             "<class 'numpy.float64'>",
#             "<class 'numpy.int32'>",
#             "<class 'numpy.int64'>"],\
#         "U must be a number. Not " + str(type(U)) + "."
#
#     # Removed the U split into two scalars because U is already a scalar.
#
#     h1 = np.array([h,0,0,0])
#     h2 = np.array([0,h,0,0])
#     h3 = np.array([0,0,h,0])
#     h4 = np.array([0,0,0,h])
#
#     # Build the dFx matrix
#
#     f_xu = np.zeros(4*1*4).reshape(4,1,4)
#
#     # d2F1[:,:,0] = np.zeros((4,1))
#
#     # d2F2[:,:,1] = np.zeros((4,1))
#
#     # f_xu_3
#
#     # f_xu[0,0,2] = ((F3(X,U)-F3(X-h1,U))-(F3(X,U-h)-F3(X-h1,U-h)))/(h**2)
#     # f_xu[1,0,2] = ((F3(X,U)-F3(X-h2,U))-(F3(X,U-h)-F3(X-h2,U-h)))/(h**2)
#     # f_xu[2,0,2] = ((F3(X,U)-F3(X-h3,U))-(F3(X,U-h)-F3(X-h3,U-h)))/(h**2)
#     # f_xu[3,0,2] = ((F3(X,U)-F3(X-h4,U))-(F3(X,U-h)-F3(X-h4,U-h)))/(h**2)
#
#     f_xu[0,0,2] = 0
#     f_xu[1,0,2] = -m2*np.sin(2*X[1])/((m1 + m2 * (np.sin(X[1])**2))**2)
#     f_xu[2,0,2] = 0
#     f_xu[3,0,2] = 0
#
#     # f_xu_4
#
#     # f_xu[0,0,3] = ((F4(X,U)-F4(X-h1,U))-(F4(X,U-h)-F4(X-h1,U-h)))/(h**2)
#     # f_xu[1,0,3] = ((F4(X,U)-F4(X-h2,U))-(F4(X,U-h)-F4(X-h2,U-h)))/(h**2)
#     # f_xu[2,0,3] = ((F4(X,U)-F4(X-h3,U))-(F4(X,U-h)-F4(X-h3,U-h)))/(h**2)
#     # f_xu[3,0,3] = ((F4(X,U)-F4(X-h4,U))-(F4(X,U-h)-F4(X-h4,U-h)))/(h**2)
#
#     f_xu[0,0,3] = 0
#     f_xu[1,0,3] = (
#             np.sin(X[1])*(m1 + m2 * (np.sin(X[1])**2))/L
#             + (np.cos(X[1])*m2*np.sin(2*X[1])/L)
#         ) / ((m1 + m2 * (np.sin(X[1])**2))**2)
#     f_xu[2,0,3] = 0
#     f_xu[3,0,3] = 0
#
#     Fxu = f_xu*dt
#     assert np.shape(Fxu)==(4,1,4) , "Fxu must be a (4,1,4) numpy array. Not shape " + str(np.shape(Fxu)) + "."
#     return(Fxu)
# def return_Fuu(X,U,dt):
#     """
#     Takes in the state vector (X), the input vector (U) and returns the discretized and linearized state matrix, Phi.
#
#     NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).
#
#     #######################
#     ##### NEED TO DO: #####
#     #######################
#
#     [ ] - Create tests that ensure that X and U are the correct dimensions.
#     [ ] - Create tests to make sure that the outputs are of the correct sizes.
#     """
#     assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
#             and np.shape(X)==(4,)), "Error with the type and shape of X ["+ return_Fuu.__name__+"()]."
#     assert str(type(U)) in ["<class 'int'>",
#             "<class 'float'>",
#             "<class 'numpy.float'>",
#             "<class 'numpy.float64'>",
#             "<class 'numpy.int32'>",
#             "<class 'numpy.int64'>"],\
#         "U must be a number. Not " + str(type(U)) + "."
#
#     # Removed the U split into two scalars because U is already a scalar.
#
#     # Build the f_uu matrix
#
#     f_uu = np.zeros(1*1*4).reshape(1,1,4)
#
#     # d2F1[0,0,0] = 0
#
#     # d2F2[0,0,1] = 0
#
#     # f_uu_3
#     # f_uu[0,0,2] = ((F3(X,U)-F3(X,U-h))-(F3(X,U-h)-F3(X,U-2*h)))/(h**2)
#
#     # f_uu_4
#     # f_uu[0,0,3] = ((F4(X,U)-F4(X,U-h))-(F4(X,U-h)-F4(X,U-2*h)))/(h**2)
#
#     Fuu = f_uu*dt
#     assert np.shape(Fuu)==(1,1,4) , "Fuu must be a (1,1,4) numpy array. Not shape " + str(np.shape(Fuu)) + "."
#     return(Fuu)
#
# def return_quadratic_dynamics_expansion(X,U,dt):
#     """
#     Takes in the input U and the the corresponding output X, as well as dt and returns two lists that contain the linearized dynamic matrices for each timestep for range(len(Time)-1).
#
#     Note that if np.shape(X)[1] = N and len(U) = M, then N = M + 1 (i.e., there is one more timestep for output than input since the initial conditions are assigned to the first state space timestep). Therefore, we only concern ourselves with the linearized dynamics of the (N-1) steps where U drives X to the next timestep (i.e., X will only go up to the N-1 step or index X[:,:-1].)
#
#     Phi is a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
#
#     B is a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
#
#     NOTE ON Fxx, Fxu, and Fuu: These are tensor values. In order to use these for the quadratic cost expansion, you must use numpy.matrix(numpy.tensordot(Fab,Vx,axes=1)) to return the proper matrix.
#
#     ### NEEDS TO BE TESTED ###
#
#     np.shape(X)[1] == len(U)+1
#
#     len(Phi) == len(U)
#     type(Phi) == list
#     len(B) == len(U)
#     type(B) == list
#
#     ##########################
#     """
#     Phi = list(
#             map(
#                 lambda X,U: return_Phi(X,U,dt),
#                 X[:,:-1].T,
#                 U
#             )
#         )
#
#     B = list(
#             map(
#                 lambda X,U: return_B(X,U,dt),
#                 X[:,:-1].T,
#                 U
#             )
#         )
#
#     Fxx = list(
#             map(
#                 lambda X,U: return_Fxx(X,U,dt),
#                 X[:,:-1].T,
#                 U
#             )
#         )
#
#     Fxu = list(
#             map(
#                 lambda X,U: return_Fxu(X,U,dt),
#                 X[:,:-1].T,
#                 U
#             )
#         )
#
#     Fuu = list(
#             map(
#                 lambda X,U: return_Fuu(X,U,dt),
#                 X[:,:-1].T,
#                 U
#             )
#         )
#     return(Phi,B,Fxx,Fxu,Fuu)
# def return_f32(X,U):
#     return(
#         (
#             (
#                 m2*L*np.cos(X[1])*(X[3]**2)
#                 - m2*gr*np.cos(2*X[1])
#                 - b2*np.sin(X[1])*X[3]/L
#             )
#             *
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#             -
#             (
#                 m2*L*np.sin(X[1])*(X[3]**2)
#                 - m2*gr*np.sin(2*X[1])/2
#                 - b1*X[2]
#                 + U
#                 + b2*np.cos(X[1])*X[3]/L
#             )
#             *
#             (
#                 m2*np.sin(2*X[1])
#             )
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
# def return_f322(X,U):
#     return(
#         (
#             (
#                 (
#                     -m2*L*np.sin(X[1])*(X[3]**2)
#                     + 2*m2*gr*np.sin(2*X[1])
#                     - b2*np.cos(X[1])*X[3]/L
#                 )
#                 *
#                 (
#                     m1 + m2*(np.sin(X[1])**2)
#                 )
#                 - (
#                     m2*L*np.sin(X[1])*(X[3]**2)
#                     - m2*gr*np.sin(2*X[1])/2
#                     - b1*X[2]
#                     + U
#                     + b2*np.cos(X[1])*X[3]/L
#                 )
#                 *
#                 (
#                     2*m2*np.cos(2*X[1])
#                 )
#             )
#             *
#             (
#                 (
#                     m1 + m2*(np.sin(X[1])**2)
#                 )**2
#             )
#             -
#             (
#                 (
#                     m2*L*np.cos(X[1])*(X[3]**2)
#                     - m2*gr*np.cos(2*X[1])
#                     - b2*np.sin(X[1])*X[3]/L
#                 )
#                 *
#                 (
#                     m1 + m2*(np.sin(X[1])**2)
#                 )
#                 -
#                 (
#                     m2*L*np.sin(X[1])*(X[3]**2)
#                     - m2*gr*np.sin(2*X[1])/2
#                     - b1*X[2]
#                     + U
#                     + b2*np.cos(X[1])*X[3]/L
#                 )
#                 *
#                 (
#                     m2*np.sin(2*X[1])
#                 )
#             )
#             *
#             (
#                 2
#                 * (
#                     m1 + m2*(np.sin(X[1])**2)
#                 )
#                 * (
#                     m2*np.sin(2*X[1])
#                 )
#             )
#         ) /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**4
#         )
#     )
# def return_f323(X,U):
#     return(
#         (
#             b1*m2*np.sin(2*X[1])
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
# def return_f324(X,U):
#     return(
#         (
#             (
#                 2*m2*L*np.cos(X[1])*X[3]
#                 - b2*np.sin(X[1])/L
#             )
#             *
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#             -
#             (
#                 2*m2*L*np.sin(X[1])*X[3]
#                 + b2*np.cos(X[1])/L
#             )
#             *
#             (
#                 m2*np.sin(2*X[1])
#             )
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
#
# def return_f33(X,U):
#     return(
#         (
#             -b1
#         )
#         /
#         (
#             m1 + m2*(np.sin(X[1])**2)
#         )
#     )
# def return_f332(X,U):
#     return(
#         (
#             b1*m2*np.sin(2*X[1])
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
# def return_f333(X,U):
#     return(
#         0
#     )
# def return_f334(X,U):
#     return(
#         0
#     )
#
# def return_f34(X,U):
#     return(
#         (
#             2*m2*L*np.sin(X[1])*X[3]
#             + b2*np.cos(X[1])/L
#         )
#         /
#         (
#             m1 + m2*(np.sin(X[1])**2)
#         )
#     )
# def return_f342(X,U):
#     return(
#         (
#             (
#                 2*m2*L*np.cos(X[1])*X[3]
#                 - b2*np.sin(X[1])/L
#             )
#             *
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#             -
#             (
#                 2*m2*L*np.sin(X[1])*X[3]
#                 + b2*np.cos(X[1])/L
#             )
#             *
#             (
#                 m2*np.sin(2*X[1])
#             )
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             ) **2
#         )
#     )
# def return_f343(X,U):
#     return(
#         0
#     )
# def return_f344(X,U):
#     return(
#             (
#                 2*m2*L*np.sin(X[1])
#             )
#             /
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#     )
#
# def return_f42(X,U):
#     return(
#         (
#             (
#                 -m2*np.cos(2*X[1])*(X[3]**2)
#                 + (m1+m2)*gr*np.cos(X[1])/L
#                 - b1*np.sin(X[1])*X[2]/L
#                 + np.sin(X[1])*U/L
#             )
#             *
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#             -
#             (
#                 -m2*np.sin(2*X[1])*(X[3]**2)/2
#                 + (m1+m2)*gr*np.sin(X[1])/L
#                 + b1*np.cos(X[1])*X[2]/L
#                 - (m1+m2)*b2*X[3]/(m2*(L**2))
#                 - np.cos(X[1])*U/L
#             )
#             *
#             (
#                 m2*np.sin(2*X[1])
#             )
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
# def return_f422(X,U):
#     return(
#         (
#             (
#                 (
#                     2*m2*np.sin(2*X[1])*(X[3]**2)
#                     - (m1+m2)*gr*np.sin(X[1])/L
#                     - b1*np.cos(X[1])*X[2]/L
#                     + np.cos(X[1])*U/L
#                 )
#                 *
#                 (
#                     m1 + m2*(np.sin(X[1])**2)
#                 )
#                 -
#                 (
#                     -m2*np.sin(2*X[1])*(X[3]**2)/2
#                     + (m1+m2)*gr*np.sin(X[1])/L
#                     + b1*np.cos(X[1])*X[2]/L
#                     - (m1+m2)*b2*X[3]/(m2*(L**2))
#                     - np.cos(X[1])*U/L
#                 )
#                 *
#                 (
#                     2*m2*np.cos(2*X[1])
#                 )
#             )
#             *
#             (
#                 (
#                     m1 + m2*(np.sin(X[1])**2)
#                 )**2
#             )
#             -
#             (
#                 (
#                     -m2*np.cos(2*X[1])*(X[3]**2)
#                     + (m1+m2)*gr*np.cos(X[1])/L
#                     - b1*np.sin(X[1])*X[2]/L
#                     + np.sin(X[1])*U/L
#                 )
#                 *
#                 (
#                     m1 + m2*(np.sin(X[1])**2)
#                 )
#                 -
#                 (
#                     -m2*np.sin(2*X[1])*(X[3]**2)/2
#                     + (m1+m2)*gr*np.sin(X[1])/L
#                     + b1*np.cos(X[1])*X[2]/L
#                     - (m1+m2)*b2*X[3]/(m2*(L**2))
#                     - np.cos(X[1])*U/L
#                 )
#                 *
#                 (
#                     m2*np.sin(2*X[1])
#                 )
#             )
#             *
#             (
#                 2
#                 * (
#                     m1 + m2*(np.sin(X[1])**2)
#                 )
#                 * (
#                     m2*np.sin(2*X[1])
#                 )
#             )
#         )
#         / (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
# def return_f423(X,U):
#     return(
#         (
#             (
#                 -b1*np.sin(X[1])/L
#             )
#             *
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#             -
#             (
#                 b1*np.cos(X[1])/L
#             )
#             *
#             (
#                 m2*np.sin(2*X[1])
#             )
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
# def return_f424(X,U):
#     return(
#         (
#             (
#                 -2*m2*np.cos(2*X[1])*X[3]
#             )
#             *
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#             -
#             (
#                 -m2*np.sin(2*X[1])*X[3]
#                 - (m1+m2)*b2/(m2*(L**2))
#             )
#             *
#             (
#                 m2*np.sin(2*X[1])
#             )
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
#
# def return_f43(X,U):
#     return(
#         (
#             b1*np.cos(X[1])/L
#         )
#         /
#         (
#             m1 + m2*(np.sin(X[1])**2)
#         )
#     )
# def return_f432(X,U):
#     return(
#         (
#             (
#                 -b1*np.sin(X[1])/L
#             )
#             *
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#             -
#             (
#                 b1*np.cos(X[1])/L
#             )
#             *
#             (
#                 m2*np.sin(2*X[1])
#             )
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
# def return_f433(X,U):
#     return(
#         0
#     )
# def return_f434(X,U):
#     return(
#         0
#     )
#
# def return_f44(X,U):
#     return(
#         (
#             -m2*np.sin(2*X[1])*X[3]
#             - (m1+m2)*b2/(m2*(L**2))
#         )
#         /
#         (
#             m1 + m2*(np.sin(X[1])**2)
#         )
#     )
# def return_f442(X,U):
#     return(
#         (
#             (
#                 -2*m2*np.cos(2*X[1])*X[3]
#             )
#             *
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )
#             -
#             (
#                 -m2*np.sin(2*X[1])*X[3]
#                 - (m1+m2)*b2/(m2*(L**2))
#             )
#             *
#             (
#                 m2*np.sin(2*X[1])
#             )
#         )
#         /
#         (
#             (
#                 m1 + m2*(np.sin(X[1])**2)
#             )**2
#         )
#     )
# def return_f443(X,U):
#     return(
#         0
#     )
# def return_f444(X,U):
#     return(
#         (
#             -m2*np.sin(2*X[1])
#         )
#         /
#         (
#             m1 + m2*(np.sin(X[1])**2)
#         )
#     )
#
# #
# # def return_Phi(X,U,dt):
# #     """
# #     Takes in the state vector, X, of shape (4,) and a number U, and outputs a matrix of shape (4,4)
# #     """
# #     assert np.shape(X)==(4,) and str(type(X))=="<class 'numpy.ndarray'>", \
# #         "X must be an numpy array of shape (4,)"
# #     assert str(type(U)) in ["<class 'int'>",
# #             "<class 'float'>",
# #             "<class 'numpy.float'>",
# #             "<class 'numpy.float64'>",
# #             "<class 'numpy.int32'>",
# #             "<class 'numpy.int64'>"],\
# #         "U must be a number. Not " + str(type(U)) + "."
# #     result = (np.eye(4)
# #         + np.matrix(
# #             [
# #             [0, 0, dt, 0],
# #             [0, 0, 0, dt],
# #             [0, return_f32(X,U)*dt, return_f33(X,U)*dt, return_f34(X,U)*dt],
# #             [0, return_f42(X,U)*dt, return_f43(X,U)*dt, return_f44(X,U)*dt]
# #             ]
# #         )
# #     )
# #
# #     assert np.shape(result)==(4,4) \
# #             and str(type(result))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
# #         "result must be a (4,4) numpy matrix. Not " + str(type(result)) + " of shape " + str(np.shape(result)) + "."
# #
# #     return(result)
# # def return_B(X,U,dt):
# #     """
# #     Takes in the state vector, X, of shape (4,) and a number U, and outputs a matrix of shape (4,1)
# #     """
# #     assert np.shape(X)==(4,) and str(type(X))=="<class 'numpy.ndarray'>", \
# #         "X must be an numpy array of shape (4,)"
# #     assert str(type(U)) in ["<class 'int'>",
# #             "<class 'float'>",
# #             "<class 'numpy.float'>",
# #             "<class 'numpy.float64'>",
# #             "<class 'numpy.int32'>",
# #             "<class 'numpy.int64'>"],\
# #         "U must be a number. Not " + str(type(U)) + "."
# #     result = (
# #         np.matrix(
# #             [
# #             [0],
# #             [0],
# #             [(dt)/(m1 + m2*(np.sin(X[1])**2))],
# #             [(-np.cos(X[1])*dt/L)/(m1 + m2*(np.sin(X[1])**2))]
# #             ]
# #         )
# #     )
# #
# #     assert np.shape(result)==(4,1) \
# #             and str(type(result))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
# #         "result must be a (4,1) numpy matrix. Not " + str(type(result)) + " of shape " + str(np.shape(result)) + "."
# #
# #     return(result)
# # def return_linearized_dynamics_matrices(X,U,dt):
# #     """
# #     Takes in the input U and the the corresponding output X, as well as dt and returns two lists that contain the linearized dynamic matrices for each timestep for range(len(Time)-1).
# #
# #     Note that if np.shape(X)[1] = N and len(U) = M, then N = M + 1 (i.e., there is one more timestep for output than input since the initial conditions are assigned to the first state space timestep). Therefore, we only concern ourselves with the linearized dynamics of the (N-1) steps where U drives X to the next timestep (i.e., X will only go up to the N-1 step or index X[:,:-1].)
# #
# #     Phi is a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
# #
# #     B is a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
# #
# #     ### NEEDS TO BE TESTED ###
# #
# #     np.shape(X)[1] == len(U)+1
# #
# #     len(Phi) == len(U)
# #     type(Phi) == list
# #     len(B) == len(U)
# #     type(B) == list
# #
# #     ##########################
# #     """
# #     Phi = list(
# #             map(
# #                 lambda X,U: return_Phi(X,U,dt),
# #                 X[:,:-1].T,
# #                 U
# #             )
# #         )
# #
# #     B = list(
# #             map(
# #                 lambda X,U: return_B(X,U,dt),
# #                 X[:,:-1].T,
# #                 U
# #             )
# #         )
# #     return(Phi,B)
