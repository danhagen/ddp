#######################################################
#######################################################
#########  Differential Dynamic Programming  ##########
########   for 1 DOF, 2 DOA Pendulum System    ########
#######################################################
#######################################################
##                                                   ##
## Course: DDP Derivation and Control System Studies ##
##                 Author: Hagen, Daniel             ##
##                                                   ##
#######################################################
#######################################################

from class_1DOF_2DOA_TT_DDP import *
from params import *
from useful_functions import *

X_o = [0,0]
DDP = DDP_1DOF_2DOA(X_o,**params)
DDP.run_ddp()
fig1 = DDP.plot_trajectory(ReturnFig=True)
FilePath = save_figures("visualizations/1DOF_2DOA_TT/DDP/","v1.0",params,ReturnPath=True,SaveAsPDF=True)
DDP.animate_trajectory(SaveAsGif=True,FileName=FilePath+"1DOF_2DOA_TT_DDP")
#
# class DDP_1DOF_2DOA:
#     def __init__(self,X_o,**params):
#         """
#         Takes in the initial position (X_o) of the system along with an optional set of parameters (params), and runs DDP to meet some desired state.
#
#         ####################################
#         ############## params ##############
#         ####################################
#
#         1) Horizon - Number of timesteps into the future we wish to program. Must be an integer. Default is 300. (NOTE: Simulation ends at t = Horizon*dt)
#
#         2) NumberOfIterations - Number of times to iterate the DDP. Must be an integer. Default is 100.
#
#         3) dt - Discrete timestep. Must be either an int, float, float32, float64, or numpy.float. Default is 0.01. (NOTE: Simulation ends at t = Horizon*dt)
#
#         4) U_o - Initial input to the system. Must be either None (meaning zero initial input to the system) or an array with shape (Horizon-1,). Default is None.
#
#         5) p_target - Target state for the system to reach. Must be a (2,1) numpy matrix. Default is numpy.matrix([[0,np.pi/2]]).T.
#
#         6) LearningRate - rate at which the system converges to the new input. Must be either an int, float, float32, float64, or numpy.float and must be between 0 and 1. Default is 0.2.
#
#         7) Q_f - Terminal cost matrix. Must be a (2,2) numpy matrix. Default is 50*numpy.matrix(numpy.eye(2)). Each element should be positive.
#
#         8) R - Running cost matrix (only one input). Must be a (2,2) numpy matrix. Default is 0.001*numpy.matrix(numpy.eye(2)). Each element should be positive.
#
#         9) PlotResults - Boolean to determine if the results of the program will be plotted. Default is False.
#
#         10) AnimateResults - Boolean to determine if the results of the program will be animated. Default is False.
#
#         11) ReturnAllResults - Boolean to determine if all results should be returned. Default is False. (NOTE: If False, the system will only return the values for the last iteration (X,U).)
#
#
#         """
#         #------------------------------------------------>
#         #----------> Possible Parameter Values ---------->
#         #------------------------------------------------>
#
#         # Horizon - Number of timesteps into the future we wish to program
#         self.Horizon = params.get("Horizon",300)
#         assert type(self.Horizon)==int,\
#             "Horizon must be an int, not "+str(type(self.Horizon))+". Default is 300."
#
#         # NumberOfIterations - Number of times to iterate the DDP
#         self.NumberOfIterations = params.get("NumberOfIterations",100)
#         assert type(self.NumberOfIterations)==int, \
#             "NumberOfIterations must be an int, not "+str(type(self.NumberOfIterations))+". Default is 100."
#
#         # dt - Discrete timestep
#         self.dt = params.get("dt",0.01)
#         assert str(type(self.dt)) in [
#                 "<class 'int'>",
#                 "<class 'float'>",
#                 "<class 'float32'>",
#                 "<class 'float64'>",
#                 "<class 'numpy.float'>"], \
#             "dt must be an int, float, float32, float64, or numpy.float not "+str(type(self.dt))+". Default is 0.01."
#
#         # U_o - Initial input to the system.
#         self.U_o = params.get("U_o",None)
#         if self.U_o is None:
#             self.U = np.zeros((2,self.Horizon-1))
#         else:
#             assert np.shape(self.U_o)==(2,self.Horizon-1), "U_o must be of shape (2,"+str(self.Horizon-1)+") not "+str(np.shape(self.U_o))+"."
#             self.U = self.U_o
#
#         # p_target - Target state for the system to reach.
#         self.p_target = params.get("p_target",np.matrix([[0,np.pi/2]]).T)
#         assert (str(type(self.p_target))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
#                 and np.shape(self.p_target)==(2,1)), \
#             "p_target must be a (2,1) numpy matrix."
#
#         # LearningRate - rate at which the system converges to the new input.
#         self.LearningRate = params.get("LearningRate",0.2)
#         assert (str(type(self.LearningRate)) in [
#                 "<class 'int'>",
#                 "<class 'float'>",
#                 "<class 'float32'>",
#                 "<class 'float64'>",
#                 "<class 'numpy.float'>"]
#                 and (0<self.LearningRate<=1)), \
#             "LearningRate must be an int, float, float32, float64, or numpy.float not "+str(type(self.LearningRate))+", and should be between 0 and 1. Default is 0.2."
#
#         # Q_f - Terminal cost matrix
#         self.Q_f = params.get("Q_f",50*np.matrix(np.eye(2)))
#         assert (str(type(self.Q_f))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
#                 and np.shape(self.Q_f)==(2,2)), \
#             "Q_f must be a (2,2) numpy matrix."
#
#         # R - Running cost scalar (only one input).
#         self.R = params.get("R",1e-3*np.eye(2))
#         assert (str(type(self.R))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
#                 and np.shape(self.R)==(2,2)), \
#             "R must be a (2,2) numpy matrix. Default is [[1e-3,0],[0,1e-3]]."
#
#         self.X_o = X_o
#         assert np.shape(self.X_o)==(2,), "X_o must have shape (2,) not "+str(np.shape(self.X_o))+"."
#         #
#         # # PlotResults - Boolean to determine if the results of the program will be plotted.
#         # PlotResults = params.get("PlotResults",False)
#         # assert type(PlotResults)==bool, "PlotResults must be either True or False (Default)."
#         #
#         # # AnimateResults - Boolean to determine if the results of the program will be animated.
#         # AnimateResults = params.get("AnimateResults",False)
#         # assert type(AnimateResults)==bool, "AnimateResults must be either True or False (Default)."
#         #
#         # # ReturnAllResults - Boolean to determine if all of the results will be returned.
#         # ReturnAllResults = params.get("ReturnAllResults",False)
#         # assert type(ReturnAllResults)==bool, "ReturnAllResults must be either True or False (Default)."
#
#     def set_Horizon(self,Horizon):
#         """
#         Horizon - Number of timesteps into the future we wish to program. Must be an integer. Default is 300. (NOTE: Simulation ends at t = Horizon*dt)
#         """
#         self.Horizon = Horizon
#         assert type(self.Horizon)==int,\
#             "Horizon must be an int, not "+str(type(self.Horizon))+". Default is 300."
#     def set_NumberOfIterations(self,NumberOfIterations):
#         """
#         NumberOfIterations - Number of times to iterate the DDP. Must be an integer. Default is 100.
#         """
#         self.NumberOfIterations = NumberOfIterations
#         assert type(self.NumberOfIterations)==int, \
#             "NumberOfIterations must be an int, not "+str(type(self.NumberOfIterations))+". Default is 100."
#     def set_dt(self,dt):
#         """
#         dt - Discrete timestep. Must be either an int, float, float32, float64, or numpy.float. Default is 0.01. (NOTE: Simulation ends at t = Horizon*dt)
#         """
#         self.dt = dt
#         assert str(type(self.dt)) in [
#                 "<class 'int'>",
#                 "<class 'float'>",
#                 "<class 'float32'>",
#                 "<class 'float64'>",
#                 "<class 'numpy.float'>"], \
#             "dt must be an int, float, float32, float64, or numpy.float not "+str(type(self.dt))+". Default is 0.01."
#     def set_U_o(self,U_o):
#         """
#         U_o - Initial input to the system. Must be either None (meaning zero initial input to the system) or an array with shape (2,Horizon-1). Default is None.
#         """
#         self.U_o = U_o
#         if self.U_o is None:
#             self.U = np.zeros((2,self.Horizon-1))
#         else:
#             assert np.shape(self.U_o)==(2,self.Horizon-1), "U_o must be of shape (2,"+str(self.Horizon-1)+") not "+str(np.shape(self.U_o))+"."
#             self.U = self.U_o
#     def set_X_o(self,X_o):
#         """
#         X_o - Initial states of the system. Must be of shape (2,).
#         """
#         self.X_o = X_o
#         assert np.shape(self.X_o)==(2,), "X_o must have shape (2,) not "+str(np.shape(self.X_o))+"."
#     def set_p_target(self,p_target):
#         """
#         p_target - Target state for the system to reach. Must be a (2,1) numpy matrix. Default is numpy.matrix([[0,np.pi/2]]).T.
#         """
#         self.p_target = p_target
#         assert (str(type(self.p_target))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
#                 and np.shape(self.p_target)==(2,1)), \
#             "p_target must be a (2,1) numpy matrix."
#     def set_LearningRate(self,LearningRate):
#         """
#         LearningRate - rate at which the system converges to the new input. Must be either an int, float, float32, float64, or numpy.float and must be between 0 and 1. Default is 0.2.
#         """
#         self.LearningRate = LearningRate
#         assert (str(type(self.LearningRate)) in [
#                 "<class 'int'>",
#                 "<class 'float'>",
#                 "<class 'float32'>",
#                 "<class 'float64'>",
#                 "<class 'numpy.float'>"]
#                 and (0<self.LearningRate<=1)), \
#             "LearningRate must be an int, float, float32, float64, or numpy.float not "+str(type(self.LearningRate))+", and should be between 0 and 1. Default is 0.2."
#     def set_Q_f(self,Q_f):
#         """
#         Q_f - Terminal cost matrix. Must be a (2,2) numpy matrix. Default is 50*numpy.matrix(numpy.eye(2)). Each element should be positive.
#         """
#         self.Q_f = Q_f
#         assert (str(type(self.Q_f))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
#                 and np.shape(self.Q_f)==(2,2)), \
#             "Q_f must be a (2,2) numpy matrix. Default is 50*numpy.matrix(numpy.eye(2))."
#     def set_R(self,R):
#         """
#         R - Running cost matrix. Must be a (2,2) numpy matrix. Default is 0.001*numpy.matrix(numpy.eye(2)). Each element should be positive.
#         """
#         self.R = R
#         assert (str(type(self.R))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
#                 and np.shape(self.R)==(2,2)), \
#             "R must be a (2,2) numpy matrix. Default is 0.001*numpy.matrix(numpy.eye(2))."
#
#     def forward_integrate_dynamics(self):
#         """
#         ICs must be a list of floats and/or ints of length 2. If ReturnX is True, the this will return an array of shape (2,len(Time)).
#
#         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         **kwargs
#         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#         UsingDegrees must be a bool. Default is True. If True, then the ICs for pendulum angle and angular velocity can be given in degrees and degrees per second, respectively.
#
#         AnimateStates must be a bool. Default is False. If True, the program will run animate_trajectory().
#
#         PlotStates must be a bool. Default is False. If True, the program will run plot the resulting states.
#
#         dt must be a number. Default is 0.01. Used with Horizon to define the time array (Time).
#
#         Horizon must be a number. Default is 300. Used with dt to define the time array (Time).
#
#         U can either be None (default) or can be an array with lenth (2,len(Time)-1). If None, then U will be chosen to be np.zeros((2,len(Time)-1))
#         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         Notes:
#
#         X1: angle of pendulum
#         X2: angular velocity of pendulum
#
#         """
#
#         self.X[0,0] = self.X_o[0]
#         self.X[1,0] = self.X_o[1]
#
#         for i in range(self.Horizon-1):
#             self.X[0,i+1] = self.X[0,i] + F1(self.X[:,i],self.U[:,i])*self.dt
#             self.X[1,i+1] = self.X[1,i] + F2(self.X[:,i],self.U[:,i])*self.dt
#
#     def return_Phi(self,x,u):
#         """
#         Takes in the state vector (X), the input vector (U) and returns the discretized and linearized state matrix, Phi.
#
#         NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).
#
#         #######################
#         ##### NEED TO DO: #####
#         #######################
#
#         [ ] - Create tests that ensure that X and U are the correct dimensions.
#         [ ] - Create tests to make sure that the outputs are of the correct sizes.
#         """
#         assert (str(type(x)) in ["<class 'numpy.ndarray'>"]
#                 and np.shape(x)==(2,)), "Error with the type and shape of x ["+ return_Phi.__name__+"()]."
#         assert (str(type(u)) in ["<class 'numpy.ndarray'>"]
#                 and np.shape(u)==(2,)), "Error with the type and shape of u ["+ return_Phi.__name__+"()]."
#
#         # Removed the U split into two scalars because U is already a scalar.
#
#         h1 = np.array([h,0])
#         h2 = np.array([0,h])
#
#         # Build the dFx matrix
#
#         dFx = np.zeros((2,2))
#
#         # dFx[0,0] = 0 # dF1/dx1⋅dx1 = (F1(x,u)-F1(x-h1,u))/h = 0
#         dFx[0,1] = 1 # dF1/dx2⋅dx2 = (F1(x,u)-F1(x-h2,u))/h = 1
#
#         # F2 is the angular acceleration of the pendulum.
#         dFx[1,0] = (F2(x,u)-F2(x-h1,u))/h
#         dFx[1,1] = (F2(x,u)-F2(x-h2,u))/h
#
#         Phi = np.matrix(np.eye(2) + dFx*self.dt)
#         assert np.shape(Phi)==(2,2) \
#             and str(type(Phi))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
#         "Phi must be a (2,2) numpy matrix. Not " + str(type(Phi)) + " of shape " + str(np.shape(Phi)) + "."
#         return(Phi)
#     def return_B(self,x,u):
#         """
#         Takes in the state vector (x), the input vector (u) and returns the discretized and linearized input matrix, B.
#
#         NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).
#
#         #######################
#         ##### NEED TO DO: #####
#         #######################
#
#         [ ] - Create tests that ensure that x and u are the correct dimensions.
#         [ ] - Create tests to make sure that the outputs are of the correct sizes.
#         """
#         assert (str(type(x)) in ["<class 'numpy.ndarray'>"]
#                 and np.shape(x)==(2,)), "Error with the type and shape of x ["+ return_B.__name__+"()]."
#         assert (str(type(u)) in ["<class 'numpy.ndarray'>"]
#                 and np.shape(u)==(2,)), "Error with the type and shape of u ["+ return_B.__name__+"()]."
#
#         # Removed the u split into two scalars because u is already a scalar.
#
#         h1 = np.array([h,0])
#         h2 = np.array([0,h])
#
#         #Build the dFu matrix
#
#         dFu = np.zeros((2,2))
#
#         dFu[0,0] = 0
#         dFu[0,1] = 0
#
#         dFu[1,0] = (F2(x,u)-F2(x,u-h1))/h
#         dFu[1,1] = (F2(x,u)-F2(x,u-h2))/h
#
#         B = np.matrix(dFu*self.dt)
#         assert np.shape(B)==(2,2) \
#                 and str(type(B))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
#             "B must be a (2,2) numpy matrix. Not " + str(type(B)) + " of shape " + str(np.shape(B)) + "."
#
#         return(B)
#     def return_linearized_dynamics_matrices(self):
#         """
#         Takes in the input U and the the corresponding output X, as well as dt and returns two lists that contain the linearized dynamic matrices for each timestep for range(len(Time)-1).
#
#         Note that if np.shape(X)[1] = N and len(U) = M, then N = M + 1 (i.e., there is one more timestep for output than input since the initial conditions are assigned to the first state space timestep). Therefore, we only concern ourselves with the linearized dynamics of the (N-1) steps where U drives X to the next timestep (i.e., X will only go up to the N-1 step or index X[:,:-1].)
#
#         Phi is a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
#
#         B is a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
#
#         ### NEEDS TO BE TESTED ###
#
#         np.shape(X)[1] == len(U)+1
#
#         len(Phi) == len(U)
#         type(Phi) == list
#         len(B) == len(U)
#         type(B) == list
#
#         ##########################
#         """
#         Phi = list(
#                 map(
#                     lambda x,u: self.return_Phi(x,u),
#                     self.X[:,:-1].T,
#                     self.U.T
#                 )
#             )
#
#         B = list(
#                 map(
#                     lambda x,u: self.return_B(x,u),
#                     self.X[:,:-1].T,
#                     self.U.T
#                 )
#             )
#         return(Phi,B)
#
#     def return_quadratic_cost_function_expansion_variables(self):
#         """
#         Takes in the input U and the the corresponding output X, as well as dt and returns lists that contain the coefficient matrices for the quadratic expansion of the cost function (l(x,u)) for each timestep for range(len(Time)-1).
#         """
#         # returns a list of length len(Time)-1, each element with shape (1,1), where n is the number of states.
#         l = list(
#                 map(
#                     lambda x,u: u[:,np.newaxis].T * self.R * u[:,np.newaxis] * self.dt,
#                     self.X[:,1:].T,
#                     self.U.T
#                 )
#             )
#
#         # returns a list of length len(Time)-1, each element with shape (n,1), where n is the number of states.
#         lx = list(
#                 map(
#                     lambda x,u: np.matrix(np.zeros((2,1)))*self.dt,
#                     self.X[:,1:].T,
#                     self.U.T
#                 )
#             )
#
#         # returns a list of length len(Time)-1, each element with shape (m,1), where n is the number of states.
#         lu = list(
#                 map(
#                     lambda x,u: self.R * u[:,np.newaxis] * self.dt,
#                     self.X[:,1:].T,
#                     self.U.T
#                 )
#             )
#
#         # returns a list of length len(Time)-1, each element with shape (m,n), where m is the number of inputs and n is the number of states.
#         lux = list(
#                 map(
#                     lambda x,u: np.matrix(np.zeros((2,2)))*self.dt,
#                     self.X[:,1:].T,
#                     self.U.T
#                 )
#             )
#
#         # returns a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
#         lxu = list(
#                 map(
#                     lambda x,u: np.matrix(np.zeros((2,2)))*self.dt,
#                     self.X[:,1:].T,
#                     self.U.T
#                 )
#             )
#
#         # returns a list of length len(Time)-1, each element with shape (m,m), where m is the number of inputs.
#         luu = list(
#                 map(
#                     lambda x,u: self.R*self.dt,
#                     self.X[:,1:].T,
#                     self.U.T
#                 )
#             )
#
#         # returns a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
#         lxx = list(
#                 map(
#                     lambda x,u: np.matrix(np.zeros((2,2)))*self.dt,
#                     self.X[:,1:].T,
#                     self.U.T
#                 )
#             )
#
#         return(l,lx,lu,lux,lxu,luu,lxx)
#
#     def return_cost_for_a_given_trial(self):
#         """
#         This takes in the state variables over time (X) and the input (U), as well as the target state (p_target), the time step (dt), and the cost matrices (Q_f and R), and output the cost of the trial.
#         """
#
#         RunningCost = 0
#         for j in range(self.Horizon-1):
#             RunningCost = RunningCost + 0.5 * self.U[:,np.newaxis,j].T * self.R * self.U[:,np.newaxis,j] * self.dt
#
#         TerminalCost = (
#             (np.matrix(self.X[:,np.newaxis,self.Horizon-1]) - self.p_target).T
#             * self.Q_f
#             * (np.matrix(self.X[:,np.newaxis,self.Horizon-1]) - self.p_target)
#         )
#
#         Cost = RunningCost[0,0] + TerminalCost[0,0]
#         return(Cost)
#
#     def run_ddp(self):
#         #------------------------------------------------>
#         #-----------> Initializing the Problem ---------->
#         #------------------------------------------------>
#
#         # TotalX and TotalU
#         TotalX = []
#         TotalU = []
#
#         # Correction array for input, not derivative of input.
#         dU = np.zeros((2,self.Horizon-1))
#
#         # Initial trajectory:
#         assert np.shape(self.X_o)==(2,), "X_o must have shape (2,) not "+str(np.shape(self.X_o))+"."
#         self.X = np.zeros((2,self.Horizon))
#         self.forward_integrate_dynamics()
#
#         V = [None]*self.Horizon # Each element must be a (2,1) matrix.
#         Vx = [None]*self.Horizon # Each element must be a (2,1) matrix.
#         Vxx = [None]*self.Horizon # Each element must be a (2,2) matrix.
#
#         Qu = [None]*self.Horizon # Each element must be a (1,1) matrix.
#         Qx = [None]*self.Horizon # Each element must be a (2,1) matrix.
#         Qux = [None]*self.Horizon # Each element must be a (2,2) matrix.
#         Qxu = [None]*self.Horizon # Each element must be a (2,2) matrix.
#         Quu = [None]*self.Horizon # Each element must be a (2,2) matrix.
#         Quu_inv = [None]*self.Horizon # Each element must be a (2,2) matrix.
#         Qxx = [None]*self.Horizon # Each element must be a (2,2) matrix.
#
#         self.TotalCost = [None]*self.NumberOfIterations
#
#         StartTime = time.time()
#         for k in range(self.NumberOfIterations):
#             #------------------------------------------------>
#             #--------> Linearization of the dynamics -------->
#             #> Quadratic Approximations of the cost function >
#             #------------------------------------------------>
#
#             Phi,B = self.return_linearized_dynamics_matrices()
#             l,lx,lu,lux,lxu,luu,lxx = self.return_quadratic_cost_function_expansion_variables()
#
#             #------------------------------------------------>
#             #--------------> Find the controls -------------->
#             #------------------------------------------------>
#
#             Vxx[self.Horizon-1]= self.Q_f
#             Vx[self.Horizon-1] = self.Q_f * (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target)
#             V[self.Horizon-1] = (
#                 (np.matrix(self.X[:,np.newaxis,self.Horizon-1]) - self.p_target).T
#                 * self.Q_f
#                 * (np.matrix(self.X[:,np.newaxis,self.Horizon-1]) - self.p_target)
#             )
#
#             #------------------------------------------------>
#             #----> Backpropagation of the Value Function ---->
#             #------------------------------------------------>
#
#             for j in reversed(range(self.Horizon-1)):
#                 Qx[j] = lx[j] + Phi[j].T * Vx[j+1]
#                 Qu[j] = lu[j] +  B[j].T * Vx[j+1]
#                 Qxu[j] = (
#                     lxu[j]
#                     + Phi[j].T * Vxx[j+1] * B[j]
#                 )
#                 Qux[j] = Qxu[j].T
#                 Quu[j] = (
#                     luu[j]
#                     + B[j].T * Vxx[j+1] * B[j]
#                 )
#                 Qxx[j] = (
#                     lxx[j]
#                     + Phi[j].T * Vxx[j+1] * Phi[j]
#                 )
#
#                 Quu_inv[j] = Quu[j]**(-1)
#
#                 Vxx[j] = Qxx[j] - Qxu[j] * Quu_inv[j] * Qux[j]
#                 Vx[j]= Qx[j] - Qxu[j] * Quu_inv[j] * Qu[j]
#                 # V[j] = l[j] + V[j+1] - 0.5 * Qu[j].T * Quu_inv[j] * Qu[j]
#                 V[j] = V[j+1] - 0.5*Qu[j].T * Quu_inv[j] * Qu[j]
#
#             #------------------------------------------------>
#             #-------------> Find the controlls -------------->
#             #------------------------------------------------>
#
#             U_new = np.zeros((2,self.Horizon-1))
#             dX = np.matrix(np.zeros((2,1)))
#             for i in range(self.Horizon-1):
#                 dU = -Quu_inv[i]*(Qu[i] + Qux[i]*dX)
#                 dX = Phi[i]*dX + B[i]*dU
#                 U_new[:,i] = (self.U[:,np.newaxis,i] + self.LearningRate*dU).squeeze()
#
#             self.U = U_new
#
#             #------------------------------------------------>
#             #-----> Simulation of the Nonlinear System ------>
#             #------------------------------------------------>
#
#             self.forward_integrate_dynamics()
#
#             self.TotalCost[k] =  self.return_cost_for_a_given_trial()
#
#             # print(
#             #     'DDP Iteration %d,  Current Cost = %f \n'
#             #     % (k+1,self.TotalCost[k])
#             # )
#     def run_ddp_for_mpc(self):
#         #------------------------------------------------>
#         #-----------> Initializing the Problem ---------->
#         #------------------------------------------------>
#
#         # TotalX and TotalU
#         TotalX = []
#         TotalU = []
#
#         # Correction array for input, not derivative of input.
#         dU = np.zeros((2,self.Horizon-1))
#
#         # Initial trajectory:
#         assert np.shape(self.X_o)==(2,), "X_o must have shape (2,) not "+str(np.shape(self.X_o))+"."
#         self.X = np.zeros((2,self.Horizon))
#         self.forward_integrate_dynamics()
#
#         V = [None]*self.Horizon # Each element must be a (2,1) matrix.
#         Vx = [None]*self.Horizon # Each element must be a (2,1) matrix.
#         Vxx = [None]*self.Horizon # Each element must be a (2,2) matrix.
#
#         Qu = [None]*self.Horizon # Each element must be a (1,1) matrix.
#         Qx = [None]*self.Horizon # Each element must be a (2,1) matrix.
#         Qux = [None]*self.Horizon # Each element must be a (2,2) matrix.
#         Qxu = [None]*self.Horizon # Each element must be a (2,2) matrix.
#         Quu = [None]*self.Horizon # Each element must be a (2,2) matrix.
#         Quu_inv = [None]*self.Horizon # Each element must be a (2,2) matrix.
#         Qxx = [None]*self.Horizon # Each element must be a (2,2) matrix.
#
#         self.TotalCost = [None]*self.NumberOfIterations
#
#         StartTime = time.time()
#         for k in range(self.NumberOfIterations):
#             #------------------------------------------------>
#             #--------> Linearization of the dynamics -------->
#             #> Quadratic Approximations of the cost function >
#             #------------------------------------------------>
#
#             Phi,B = self.return_linearized_dynamics_matrices()
#             l,lx,lu,lux,lxu,luu,lxx = self.return_quadratic_cost_function_expansion_variables()
#
#             #------------------------------------------------>
#             #--------------> Find the controls -------------->
#             #------------------------------------------------>
#
#             Vxx[self.Horizon-1]= self.Q_f
#             Vx[self.Horizon-1] = self.Q_f * (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target)
#             V[self.Horizon-1] = (
#                 (np.matrix(self.X[:,np.newaxis,self.Horizon-1]) - self.p_target).T
#                 * self.Q_f
#                 * (np.matrix(self.X[:,np.newaxis,self.Horizon-1]) - self.p_target)
#             )
#
#             #------------------------------------------------>
#             #----> Backpropagation of the Value Function ---->
#             #------------------------------------------------>
#
#             for j in reversed(range(self.Horizon-1)):
#                 Qx[j] = lx[j] + Phi[j].T * Vx[j+1]
#                 Qu[j] = lu[j] +  B[j].T * Vx[j+1]
#                 Qxu[j] = (
#                     lxu[j]
#                     + Phi[j].T * Vxx[j+1] * B[j]
#                 )
#                 Qux[j] = Qxu[j].T
#                 Quu[j] = (
#                     luu[j]
#                     + B[j].T * Vxx[j+1] * B[j]
#                 )
#                 Qxx[j] = (
#                     lxx[j]
#                     + Phi[j].T * Vxx[j+1] * Phi[j]
#                 )
#
#                 Quu_inv[j] = Quu[j]**(-1)
#
#                 Vxx[j] = Qxx[j] - Qxu[j] * Quu_inv[j] * Qux[j]
#                 Vx[j]= Qx[j] - Qxu[j] * Quu_inv[j] * Qu[j]
#                 # V[j] = l[j] + V[j+1] - 0.5 * Qu[j].T * Quu_inv[j] * Qu[j]
#                 V[j] = V[j+1] - 0.5*Qu[j].T * Quu_inv[j] * Qu[j]
#
#             #------------------------------------------------>
#             #-------------> Find the controlls -------------->
#             #------------------------------------------------>
#             if k==self.NumberOfIterations-1:
#                 dX = np.matrix(np.zeros((2,1)))
#                 dU = -Quu_inv[0]*(Qu[0] + Qux[0]*dX)
#                 self.U[0,:] = (self.U[0,0] + self.LearningRate*dU[0,0])*np.ones((self.Horizon-1,))
#                 self.U[1,:] = (self.U[1,0] + self.LearningRate*dU[1,0])*np.ones((self.Horizon-1,))
#                 self.next_X = np.zeros((2,1))
#                 self.next_X[0,0] = self.X_o[0] + F1(self.X_o,self.U[:,0])*self.dt
#                 self.next_X[1,0] = self.X_o[1] + F2(self.X_o,self.U[:,0])*self.dt
#             else:
#                 U_new = np.zeros((2,self.Horizon-1))
#                 dX = np.matrix(np.zeros((2,1)))
#                 for i in range(self.Horizon-1):
#                     dU = -Quu_inv[i]*(Qu[i] + Qux[i]*dX)
#                     dX = Phi[i]*dX + B[i]*dU
#                     U_new[:,i] = (self.U[:,np.newaxis,i] + self.LearningRate*dU).squeeze()
#
#                 self.U = U_new
#
#                 #------------------------------------------------>
#                 #-----> Simulation of the Nonlinear System ------>
#                 #------------------------------------------------>
#
#                 self.forward_integrate_dynamics()
#
#                 self.TotalCost[k] =  self.return_cost_for_a_given_trial()
#
#                 # print(
#                 #     'DDP Iteration %d,  Current Cost = %f \n'
#                 #     % (k+1,self.TotalCost[k])
#                 # )
#
#     def return_time_array(self):
#         Endtime = self.Horizon*self.dt
#         Time = np.arange(0,Endtime,self.dt)
#         return(Time)
#     def plot_trajectory(self,ReturnFig=False):
#         Time = self.return_time_array()
#
#         assert type(ReturnFig)==bool, "ReturnFig must be either true or false (default)."
#
#         fig = plt.figure(figsize=(15,10))
#         plt.suptitle("Cart Pole Control via DDP",fontsize=16)
#
#         # ax1 = plt.subplot2grid((3,2),(0,0),colspan=2)
#         ax1 = plt.subplot(322)
#         ax1.plot(Time[:-1],self.U[0,:],'r')
#         ax1.plot(Time[:-1],self.U[1,:],'g')
#         ax1.plot(
#             [Time[0],Time[-1]],
#             [0]*2,
#             'k--',
#             linewidth=2
#         )
#         ax1.set_xlabel('Time (s)')
#         ax1.set_ylabel('Tendon Tension (N)')
#         if max(abs(self.U[0,:] - self.U[0,0]))<1e-7 and max(abs(self.U[1,:] - self.U[1,0]))<1e-7:
#             ax1.set_ylim([min(self.U[:,0]) - 5,max(self.U[:,0]) + 5])
#
#         ax2 = plt.subplot(323)
#         ax2.plot(
#             [Time[0],Time[-1]],
#             [180*self.p_target[0,0]/np.pi]*2,
#             'k--',
#             linewidth=2
#         )
#         ax2.plot(Time,180*self.X[0,:]/np.pi,'b')
#         ax2.set_xlabel('Time (s)')
#         ax2.set_ylabel('Angle (deg)')
#         if max(abs(180*self.X[0,:]/np.pi - 180*self.X[0,0]/np.pi))<1e-7:
#             ax2.set_ylim([180*self.X[0,0]/np.pi - 5,180*self.X[0,0]/np.pi + 5])
#
#         ax3 = plt.subplot(324)
#         ax3.plot(
#             [Time[0],Time[-1]],
#             [180*self.p_target[1,0]/np.pi]*2,
#             'k--',
#             linewidth=2
#         )
#         ax3.plot(Time,180*self.X[1,:]/np.pi,'b--')
#         ax3.set_xlabel('Time (s)')
#         ax3.set_ylabel('Angular Velocity (deg/s)')
#         if max(abs(180*self.X[1,:]/np.pi-180*self.X[1,0]/np.pi))<1e-7:
#             ax3.set_ylim([180*self.X[1,0]/np.pi-1,180*self.X[1,0]/np.pi+1])
#
#         ax0 = plt.subplot(321)
#         Pendulum_Width = 0.01*L1
#         Pendulum_Length = L1
#
#         Ground = plt.Rectangle(
#                     (-52*Pendulum_Width/4,-Pendulum_Length/4),
#                     52*Pendulum_Width/4,
#                     Pendulum_Length/2,
#                     Color='#4682b4')
#         ax0.add_patch(Ground)
#
#
#         Pendulum, = ax0.plot(
#                         [
#                             0,
#                             Pendulum_Length*np.sin((30*np.pi/180))
#                         ],
#                         [
#                             0,
#                             -Pendulum_Length*np.cos((30*np.pi/180))
#                         ],
#                         Color='0.50',
#                         lw = 10,
#                         solid_capstyle='round'
#                         )
#
#         Pendulum_neutral, = ax0.plot(
#                         [
#                             0,
#                             0
#                         ],
#                         [
#                             0,
#                             -Pendulum_Length
#                         ],
#                         Color='k',
#                         lw = 1,
#                         linestyle='--'
#                         )
#
#         Angle_indicator, = ax0.plot(
#                         Pendulum_Length*np.sin(
#                             np.linspace(0.05*(30*np.pi/180),0.95*(30*np.pi/180),20)
#                             ),
#                         -Pendulum_Length*np.cos(
#                             np.linspace(0.05*(30*np.pi/180),0.95*(30*np.pi/180),20)
#                             ),
#                         Color='b',
#                         lw = 2,
#                         solid_capstyle = 'round'
#                         )
#         k = 0.075*Pendulum_Length
#         Angle_indicator_arrow, = ax0.plot(
#                         Pendulum_Length*np.sin(0.95*(30*np.pi/180))
#                         + [
#                             -k*np.sin((120*np.pi/180) - 0.95*(30*np.pi/180)),
#                             0,
#                             -k*np.sin((60*np.pi/180) - 0.95*(30*np.pi/180))
#                         ],
#                         -Pendulum_Length*np.cos(0.95*(30*np.pi/180))
#                         + [
#                             -k*np.cos((120*np.pi/180) - 0.95*(30*np.pi/180)),
#                             0,
#                             -k*np.cos((60*np.pi/180) - 0.95*(30*np.pi/180))
#                         ],
#                         Color='b',
#                         lw = 2,
#                         solid_capstyle='round'
#                         )
#         Angle_damping_indicator, = ax0.plot(
#                         0.50*Pendulum_Length*np.sin(
#                                 np.linspace(
#                                     0.45*(30*np.pi/180),
#                                     1.55*(30*np.pi/180),
#                                     20
#                                 )
#                             ),
#                         -0.50*Pendulum_Length*np.cos(
#                                 np.linspace(
#                                     0.45*(30*np.pi/180),
#                                     1.55*(30*np.pi/180),
#                                     20
#                                 )
#                             ),
#                         Color='#ffa500',
#                         lw = 2,
#                         solid_capstyle = 'round'
#                         )
#         Angle_damping_indicator_arrow, = ax0.plot(
#                         0.50*Pendulum_Length*np.sin(0.45*(30*np.pi/180))
#                         + [
#                             k*np.sin(0.45*(30*np.pi/180) + (60*np.pi/180)),
#                             0,
#                             k*np.sin(0.45*(30*np.pi/180) + (120*np.pi/180))
#                         ],
#                         -0.50*Pendulum_Length*np.cos(0.45*(30*np.pi/180))
#                         + [
#                             -k*np.cos(0.45*(30*np.pi/180) + (60*np.pi/180)),
#                             0,
#                             -k*np.cos(0.45*(30*np.pi/180) + (120*np.pi/180))
#                         ],
#                         Color='#ffa500',
#                         lw = 2,
#                         solid_capstyle='round'
#                         )
#
#         tau1_indicator, = ax0.plot(
#                         0.75*Pendulum_Length*np.sin(
#                                 np.linspace(
#                                     1.05*(30*np.pi/180),
#                                     1.05*(30*np.pi/180)+(45*np.pi/180),
#                                     20
#                                 )
#                             ),
#                         -0.75*Pendulum_Length*np.cos(
#                                 np.linspace(
#                                     1.05*(30*np.pi/180),
#                                     1.05*(30*np.pi/180)+(45*np.pi/180),
#                                     20
#                                 )
#                             ),
#                         Color='r',
#                         lw = 2,
#                         solid_capstyle = 'round'
#                         )
#         tau1_indicator_arrow, = ax0.plot(
#                         0.75*Pendulum_Length*np.sin(1.05*(30*np.pi/180)+(45*np.pi/180))
#                         + [
#                             -k*np.sin((120*np.pi/180) - 1.05*(30*np.pi/180)-(45*np.pi/180)),
#                             0,
#                             -k*np.sin((60*np.pi/180) - 1.05*(30*np.pi/180)-(45*np.pi/180))
#                         ],
#                         -0.75*Pendulum_Length*np.cos(1.05*(30*np.pi/180)+(45*np.pi/180))
#                         + [
#                             -k*np.cos((120*np.pi/180) - 1.05*(30*np.pi/180)-(45*np.pi/180)),
#                             0,
#                             -k*np.cos((60*np.pi/180) - 1.05*(30*np.pi/180)-(45*np.pi/180))
#                         ],
#                         Color='r',
#                         lw = 2,
#                         solid_capstyle='round'
#                         )
#
#         tau2_indicator, = ax0.plot(
#                         0.75*Pendulum_Length*np.sin(
#                                 np.linspace(
#                                     0.95*(30*np.pi/180)-(45*np.pi/180),
#                                     0.95*(30*np.pi/180),
#                                     20
#                                 )
#                             ),
#                         -0.75*Pendulum_Length*np.cos(
#                                 np.linspace(
#                                     0.95*(30*np.pi/180)-(45*np.pi/180),
#                                     0.95*(30*np.pi/180),
#                                     20
#                                 )
#                             ),
#                         Color='g',
#                         lw = 2,
#                         solid_capstyle = 'round'
#                         )
#         tau2_indicator_arrow, = ax0.plot(
#                         0.75*Pendulum_Length*np.sin(0.95*(30*np.pi/180)-(45*np.pi/180))
#                         + [
#                             k*np.sin((15*np.pi/180) + 0.95*(30*np.pi/180)),
#                             0,
#                             k*np.sin((75*np.pi/180) + 0.95*(30*np.pi/180))
#                         ],
#                         -0.75*Pendulum_Length*np.cos(0.95*(30*np.pi/180)-(45*np.pi/180))
#                         + [
#                             -k*np.cos((15*np.pi/180) + 0.95*(30*np.pi/180)),
#                             0,
#                             -k*np.cos((75*np.pi/180) + 0.95*(30*np.pi/180))
#                         ],
#                         Color='g',
#                         lw = 2,
#                         solid_capstyle='round'
#                         )
#
#
#         Pendulum_Attachment = plt.Circle((0,0),50*Pendulum_Width/4,Color='#4682b4')
#         ax0.add_patch(Pendulum_Attachment)
#
#         Pendulum_Rivet, = ax0.plot(
#             [0],
#             [0],
#             c='k',
#             marker='o',
#             lw=2
#             )
#
#         ax0.get_xaxis().set_ticks([])
#         ax0.get_yaxis().set_ticks([])
#         ax0.set_frame_on(True)
#         ax0.set_xlim([-1.60*Pendulum_Length,2.00*Pendulum_Length])
#         ax0.set_ylim([-1.10*Pendulum_Length,0.30*Pendulum_Length])
#
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         # ax0.text(0.05, 0.95, r"$b_1$ = " + str(b1) + "\n" + r"$b_2$ = " + str(b2), transform=ax0.transAxes, fontsize=14,
#         # verticalalignment='top', bbox=props)
#         ax0.legend(
#             (Angle_damping_indicator,tau1_indicator,tau2_indicator),
#             (r"$b_1\dot{\theta}$", r"$R_1(\theta)u_1$", r"$R_2(\theta)u_2$"),
#             loc='upper left',
#             facecolor='wheat',
#             framealpha=0.5,
#             title="Torques")
#         ax0.set_aspect('equal')
#
#         ax4 = plt.subplot(3,2,(5,6))
#         ax4.plot(self.TotalCost,linewidth=2)
#         ax4.set_xlabel('Iterations',fontsize=16)
#         ax4.set_ylabel('Cost',fontsize=16)
#
#         if ReturnFig==True:
#             return(fig)
#         else:
#             plt.show()
#
#     def animate_trajectory(self,**kwargs):
#         assert hasattr(self,"X"), "Run DDP before plotting."
#         Time = self.return_time_array()
#
#         SaveAsGif = kwargs.get("SaveAsGif",False)
#         assert type(SaveAsGif)==bool, "SaveAsGif must be either True or False (Default)."
#
#         FileName = kwargs.get("FileName","1DOF_2DOA_TT")
#         assert type(FileName)==str,"FileName must be a str."
#
#             # Angles must be in degrees for animation
#
#         X1d = self.X[0,:]*(180/np.pi)
#         X2d = self.X[1,:]*(180/np.pi)
#
#
#         fig = plt.figure(figsize=(12,10))
#         ax0 = plt.subplot2grid((2,4),(0,0),colspan=2) # animation
#         ax1 = plt.subplot2grid((2,4),(0,2),colspan=2) # input
#         ax2 = plt.subplot2grid((2,4),(1,0),colspan=2) # pendulum angle
#         ax4 = plt.subplot2grid((2,4),(1,2),colspan=2) # pendulum angular velocity
#
#         plt.suptitle("Cart-Pendulum Example",Fontsize=28,y=0.95)
#         Pendulum_Width = 0.01*L1
#         Pendulum_Length = L1
#
#         Ground = plt.Rectangle(
#                     (-52*Pendulum_Width/4,-Pendulum_Length/4),
#                     52*Pendulum_Width/4,
#                     Pendulum_Length/2,
#                     Color='#4682b4')
#         ax0.add_patch(Ground)
#
#
#         Pendulum, = ax0.plot(
#                         [
#                             0,
#                             Pendulum_Length*np.sin(self.X[0,0])
#                         ],
#                         [
#                             0,
#                             -Pendulum_Length*np.cos(self.X[0,0])
#                         ],
#                         Color='0.50',
#                         lw = 10,
#                         solid_capstyle='round'
#                         )
#
#         max_tau = self.U.max()
#         if max_tau==0: max_tau=1
#
#         k = 0.075*Pendulum_Length
#         tau1_indicator, = ax0.plot(
#                         0.75*Pendulum_Length*np.sin(
#                                 np.linspace(
#                                     1.05*self.X[0,0],
#                                     1.05*self.X[0,0] + (45*np.pi/180)*self.U[0,0]/max_tau,
#                                     20
#                                 )
#                             ),
#                         -0.75*Pendulum_Length*np.cos(
#                                 np.linspace(
#                                     1.05*self.X[0,0],
#                                     1.05*self.X[0,0] + (45*np.pi/180)*self.U[0,0]/max_tau,
#                                     20
#                                 )
#                             ),
#                         Color='r',
#                         lw = 2,
#                         solid_capstyle = 'round'
#                         )
#         tau1_indicator_arrow, = ax0.plot(
#                         0.75*Pendulum_Length*np.sin(1.05*self.X[0,0] + (45*np.pi/180)*self.U[0,0]/max_tau)
#                         + [
#                             -k*np.sin((120*np.pi/180) - 1.05*self.X[0,0] - (45*np.pi/180)*self.U[0,0]/max_tau),
#                             0,
#                             -k*np.sin((60*np.pi/180) - 1.05*self.X[0,0] - (45*np.pi/180)*self.U[0,0]/max_tau)
#                         ],
#                         -0.75*Pendulum_Length*np.cos(1.05*self.X[0,0] + (45*np.pi/180)*self.U[0,0]/max_tau)
#                         + [
#                             -k*np.cos((120*np.pi/180) - 1.05*self.X[0,0] - (45*np.pi/180)*self.U[0,0]/max_tau),
#                             0,
#                             -k*np.cos((60*np.pi/180) - 1.05*self.X[0,0] - (45*np.pi/180)*self.U[0,0]/max_tau)
#                         ],
#                         Color='r',
#                         lw = 2,
#                         solid_capstyle='round'
#                         )
#
#         tau2_indicator, = ax0.plot(
#                         0.75*Pendulum_Length*np.sin(
#                                 np.linspace(
#                                     0.95*self.X[0,0]-(45*np.pi/180)*self.U[1,0]/max_tau,
#                                     0.95*self.X[0,0],
#                                     20
#                                 )
#                             ),
#                         -0.75*Pendulum_Length*np.cos(
#                                 np.linspace(
#                                     0.95*self.X[0,0]-(45*np.pi/180)*self.U[1,0]/max_tau,
#                                     0.95*self.X[0,0],
#                                     20
#                                 )
#                             ),
#                         Color='g',
#                         lw = 2,
#                         solid_capstyle = 'round'
#                         )
#         tau2_indicator_arrow, = ax0.plot(
#                         0.75*Pendulum_Length*np.sin(0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau)
#                         + [
#                             k*np.sin((60*np.pi/180) + 0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau),
#                             0,
#                             k*np.sin((120*np.pi/180) + 0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau)
#                         ],
#                         -0.75*Pendulum_Length*np.cos(0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau)
#                         + [
#                             -k*np.cos((60*np.pi/180) + 0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau),
#                             0,
#                             -k*np.cos((120*np.pi/180) + 0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau)
#                         ],
#                         Color='g',
#                         lw = 2,
#                         solid_capstyle='round'
#                         )
#
#
#         Pendulum_Attachment = plt.Circle((0,0),50*Pendulum_Width/4,Color='#4682b4')
#         ax0.add_patch(Pendulum_Attachment)
#
#         Pendulum_Rivet, = ax0.plot(
#             [0],
#             [0],
#             c='k',
#             marker='o',
#             lw=2
#             )
#
#         ax0.get_xaxis().set_ticks([])
#         ax0.get_yaxis().set_ticks([])
#         ax0.set_frame_on(True)
#         ax0.set_xlim([-1.10*Pendulum_Length,1.10*Pendulum_Length])
#         ax0.set_ylim([-1.10*Pendulum_Length,1.10*Pendulum_Length])
#         ax0.set_aspect('equal')
#
#
#         TimeStamp = ax0.text(
#             0,
#             0.75*Pendulum_Length,
#             "Time: "+str(Time[0])+" s",
#             color='0.50',
#             fontsize=16,
#             horizontalalignment='center'
#         )
#
#         #Input
#
#         Input1, = ax1.plot([0],[self.U[0,0]],color = 'r')
#         Input2, = ax1.plot([0],[self.U[1,0]],color = 'g')
#         ax1.set_xlim(0,Time[-1])
#         ax1.set_xticks(list(np.linspace(0,Time[-1],5)))
#         ax1.set_xticklabels([str(0),'','','',str(Time[-1])])
#         if max(abs(self.U[0,:] - self.U[0,0]))<1e-7 and max(abs(self.U[1,:] - self.U[1,0]))<1e-7:
#             ax1.set_ylim([min(self.U[:,0]) - 5,max(self.U[:,0]) + 5])
#         else:
#             RangeU = self.U.max()-self.U.min()
#             ax1.set_ylim([self.U.min()-0.1*RangeU,self.U.max()+0.1*RangeU])
#
#         ax1.spines['right'].set_visible(False)
#         ax1.spines['top'].set_visible(False)
#         ax1.set_title("Tendon Tensions (N)",fontsize=16,fontweight = 4,color = 'k',y = 0.95)
#
#         #Pendulum Angle
#
#         Angle, = ax2.plot([0],[X1d[0]],color = 'k')
#         ax2.set_xlim(0,Time[-1])
#         ax2.set_xticks(list(np.linspace(0,Time[-1],5)))
#         ax2.set_xticklabels([str(0),'','','',str(Time[-1])])
#         if max(abs(X1d-X1d[0]))<1e-7:
#             ax2.set_ylim([X1d[0]-2,X1d[0]+2])
#         else:
#             RangeX1d= max(X1d)-min(X1d)
#             ax2.set_ylim([min(X1d)-0.1*RangeX1d,max(X1d)+0.1*RangeX1d])
#         ax2.spines['right'].set_visible(False)
#         ax2.spines['top'].set_visible(False)
#         ax2.set_title("Angle (deg)",fontsize=16,fontweight = 4,color = 'k',y = 0.95)
#
#         # Angular Velocity
#
#         AngularVelocity, = ax4.plot([0],[X2d[0]],color='k',linestyle='--')
#         ax4.set_xlim(0,Time[-1])
#         ax4.set_xticks(list(np.linspace(0,Time[-1],5)))
#         ax4.set_xticklabels([str(0),'','','',str(Time[-1])])
#         if max(abs(X2d-X2d[0]))<1e-7:
#             ax4.set_ylim([X2d[0]-2,X2d[0]+2])
#         else:
#             RangeX2d= max(X2d)-min(X2d)
#             ax4.set_ylim([min(X2d)-0.1*RangeX2d,max(X2d)+0.1*RangeX2d])
#         ax4.spines['right'].set_visible(False)
#         ax4.spines['top'].set_visible(False)
#         ax4.set_title("Angular Velocity (deg/s)",fontsize=16,fontweight = 4,color = 'k',y = 0.95)
#
#         def animate(i):
#             Pendulum.set_xdata([0,Pendulum_Length*np.sin(self.X[0,i])])
#             Pendulum.set_ydata([0,
#                                 -Pendulum_Length*np.cos(self.X[0,i])])
#             tau1_indicator.set_xdata(
#                 0.75*Pendulum_Length*np.sin(
#                     np.linspace(
#                         1.05*self.X[0,i],
#                         1.05*self.X[0,i] + (45*np.pi/180)*self.U[0,i]/max_tau,
#                         20
#                     )
#                 )
#             )
#             tau1_indicator.set_ydata(
#                 -0.75*Pendulum_Length*np.cos(
#                     np.linspace(
#                         1.05*self.X[0,i],
#                         1.05*self.X[0,i] + (45*np.pi/180)*self.U[0,i]/max_tau,
#                         20
#                     )
#                 )
#             )
#
#             tau1_indicator_arrow.set_xdata(
#                 0.75*Pendulum_Length*np.sin(1.05*self.X[0,i] + (45*np.pi/180)*self.U[0,i]/max_tau)
#                 + [
#                     -k*np.sin((120*np.pi/180) - 1.05*self.X[0,i] - (45*np.pi/180)*self.U[0,i]/max_tau),
#                     0,
#                     -k*np.sin((60*np.pi/180) - 1.05*self.X[0,i] - (45*np.pi/180)*self.U[0,i]/max_tau)
#                 ]
#             )
#             tau1_indicator_arrow.set_ydata(
#                 -0.75*Pendulum_Length*np.cos(1.05*self.X[0,i] + (45*np.pi/180)*self.U[0,i]/max_tau)
#                 + [
#                     -k*np.cos((120*np.pi/180) - 1.05*self.X[0,i] - (45*np.pi/180)*self.U[0,i]/max_tau),
#                     0,
#                     -k*np.cos((60*np.pi/180) - 1.05*self.X[0,i] - (45*np.pi/180)*self.U[0,i]/max_tau)
#                 ]
#             )
#
#             tau2_indicator.set_xdata(
#                 0.75*Pendulum_Length*np.sin(
#                     np.linspace(
#                         0.95*self.X[0,i]-(45*np.pi/180)*self.U[1,i]/max_tau,
#                         0.95*self.X[0,i],
#                         20
#                     )
#                 )
#             )
#             tau2_indicator.set_ydata(
#                 -0.75*Pendulum_Length*np.cos(
#                     np.linspace(
#                         0.95*self.X[0,i]-(45*np.pi/180)*self.U[1,i]/max_tau,
#                         0.95*self.X[0,i],
#                         20
#                     )
#                 )
#             )
#
#             tau2_indicator_arrow.set_xdata(
#                 0.75*Pendulum_Length*np.sin(0.95*self.X[0,i] - (45*np.pi/180)*self.U[1,i]/max_tau)
#                 + [
#                     k*np.sin((60*np.pi/180) + 0.95*self.X[0,i] - (45*np.pi/180)*self.U[1,i]/max_tau),
#                     0,
#                     k*np.sin((120*np.pi/180) + 0.95*self.X[0,i] - (45*np.pi/180)*self.U[1,i]/max_tau)
#                 ]
#             )
#             tau2_indicator_arrow.set_ydata(
#                 -0.75*Pendulum_Length*np.cos(0.95*self.X[0,i] - (45*np.pi/180)*self.U[1,i]/max_tau)
#                 + [
#                     -k*np.cos((60*np.pi/180) + 0.95*self.X[0,i] - (45*np.pi/180)*self.U[1,i]/max_tau),
#                     0,
#                     -k*np.cos((120*np.pi/180) + 0.95*self.X[0,i] - (45*np.pi/180)*self.U[1,i]/max_tau)
#                 ]
#             )
#
#
#             TimeStamp.set_text("Time: "+"{:.2f}".format(Time[i])+" s",)
#
#             Input1.set_xdata(Time[:i])
#             Input1.set_ydata(self.U[0,:i])
#
#             Input2.set_xdata(Time[:i])
#             Input2.set_ydata(self.U[1,:i])
#
#             Angle.set_xdata(Time[:i])
#             Angle.set_ydata(X1d[:i])
#
#             AngularVelocity.set_xdata(Time[:i])
#             AngularVelocity.set_ydata(X2d[:i])
#
#             return Pendulum,tau1_indicator,tau1_indicator_arrow,tau2_indicator,tau2_indicator_arrow,Input1,Input2,Angle,AngularVelocity,TimeStamp,
#
#         # Init only required for blitting to give a clean slate.
#         def init():
#             Ground = plt.Rectangle(
#                         (-52*Pendulum_Width/4,-Pendulum_Length/4),
#                         52*Pendulum_Width/4,
#                         Pendulum_Length/2,
#                         Color='#4682b4')
#             ax0.add_patch(Ground)
#
#
#             Pendulum, = ax0.plot(
#                             [
#                                 0,
#                                 Pendulum_Length*np.sin(self.X[0,0])
#                             ],
#                             [
#                                 0,
#                                 -Pendulum_Length*np.cos(self.X[0,0])
#                             ],
#                             Color='0.50',
#                             lw = 10,
#                             solid_capstyle='round'
#                             )
#
#             tau1_indicator, = ax0.plot(
#                             0.75*Pendulum_Length*np.sin(
#                                     np.linspace(
#                                         1.05*self.X[0,0],
#                                         1.05*self.X[0,0] + (45*np.pi/180)*self.U[0,0]/max_tau,
#                                         20
#                                     )
#                                 ),
#                             -0.75*Pendulum_Length*np.cos(
#                                     np.linspace(
#                                         1.05*self.X[0,0],
#                                         1.05*self.X[0,0] + (45*np.pi/180)*self.U[0,0]/max_tau,
#                                         20
#                                     )
#                                 ),
#                             Color='r',
#                             lw = 2,
#                             solid_capstyle = 'round'
#                             )
#             tau1_indicator_arrow, = ax0.plot(
#                             0.75*Pendulum_Length*np.sin(1.05*self.X[0,0] + (45*np.pi/180)*self.U[0,0]/max_tau)
#                             + [
#                                 -k*np.sin((120*np.pi/180) - 1.05*self.X[0,0] - (45*np.pi/180)*self.U[0,0]/max_tau),
#                                 0,
#                                 -k*np.sin((60*np.pi/180) - 1.05*self.X[0,0] - (45*np.pi/180)*self.U[0,0]/max_tau)
#                             ],
#                             -0.75*Pendulum_Length*np.cos(1.05*self.X[0,0] + (45*np.pi/180)*self.U[0,0]/max_tau)
#                             + [
#                                 -k*np.cos((120*np.pi/180) - 1.05*self.X[0,0] - (45*np.pi/180)*self.U[0,0]/max_tau),
#                                 0,
#                                 -k*np.cos((60*np.pi/180) - 1.05*self.X[0,0] - (45*np.pi/180)*self.U[0,0]/max_tau)
#                             ],
#                             Color='r',
#                             lw = 2,
#                             solid_capstyle='round'
#                             )
#
#             tau2_indicator, = ax0.plot(
#                             0.75*Pendulum_Length*np.sin(
#                                     np.linspace(
#                                         0.95*self.X[0,0]-(45*np.pi/180)*self.U[1,0]/max_tau,
#                                         0.95*self.X[0,0],
#                                         20
#                                     )
#                                 ),
#                             -0.75*Pendulum_Length*np.cos(
#                                     np.linspace(
#                                         0.95*self.X[0,0]-(45*np.pi/180)*self.U[1,0]/max_tau,
#                                         0.95*self.X[0,0],
#                                         20
#                                     )
#                                 ),
#                             Color='g',
#                             lw = 2,
#                             solid_capstyle = 'round'
#                             )
#             tau2_indicator_arrow, = ax0.plot(
#                             0.75*Pendulum_Length*np.sin(0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau)
#                             + [
#                                 k*np.sin((60*np.pi/180) + 0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau),
#                                 0,
#                                 k*np.sin((120*np.pi/180) + 0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau)
#                             ],
#                             -0.75*Pendulum_Length*np.cos(0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau)
#                             + [
#                                 -k*np.cos((60*np.pi/180) + 0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau),
#                                 0,
#                                 -k*np.cos((120*np.pi/180) + 0.95*self.X[0,0] - (45*np.pi/180)*self.U[1,0]/max_tau)
#                             ],
#                             Color='g',
#                             lw = 2,
#                             solid_capstyle='round'
#                             )
#
#
#             Pendulum_Attachment = plt.Circle((0,0),50*Pendulum_Width/4,Color='#4682b4')
#             ax0.add_patch(Pendulum_Attachment)
#
#             Pendulum_Rivet, = ax0.plot(
#                 [0],
#                 [0],
#                 c='k',
#                 marker='o',
#                 lw=2
#                 )
#
#             TimeStamp = ax0.text(
#                 0,
#                 0.75*Pendulum_Length,
#                 "Time: "+"{:.2f}".format(Time[0])+" s",
#                 color='0.50',
#                 fontsize=16,
#                 horizontalalignment='center'
#             )
#
#             #Input
#
#             Input1, = ax1.plot([0],[self.U[0,0]],color = 'r')
#             Input2, = ax1.plot([0],[self.U[1,0]],color = 'g')
#
#             #Pendulum Angle
#
#             Angle, = ax2.plot([0],[X1d[0]],color = 'k')
#
#             # Angular Velocity
#
#             AngularVelocity, = ax4.plot([0],[X2d[0]],color='k',linestyle='--')
#
#
#             Ground.set_visible(True)
#             Pendulum.set_visible(False)
#             tau1_indicator.set_visible(False)
#             tau1_indicator_arrow.set_visible(False)
#             tau2_indicator.set_visible(False)
#             tau2_indicator_arrow.set_visible(False)
#             Pendulum_Attachment.set_visible(False)
#             Pendulum_Rivet.set_visible(False)
#             TimeStamp.set_visible(False)
#             Input1.set_visible(False)
#             Input2.set_visible(False)
#             Angle.set_visible(False)
#             AngularVelocity.set_visible(False)
#
#             return Ground,Pendulum,tau1_indicator,tau1_indicator_arrow,tau2_indicator,tau2_indicator_arrow,Pendulum_Attachment,Pendulum_Rivet,TimeStamp,Input1,Input2,Angle,AngularVelocity,
#
#         dt = Time[1]-Time[0]
#         if dt <= 0.0001:
#             framestep=2000
#         elif dt <= 0.001:
#             framestep=200
#         elif dt <= 0.01:
#             framestep=10
#         else:
#             framestep=5
#         ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(Time)-1,framestep),init_func=init, blit=False)
#         if SaveAsGif==True:
#             ani.save(FileName+'.gif', writer='imagemagick', fps=10)
#         plt.show()
#
#     def return_all_params(self):
#         return(
#             {
#                 "States" : self.X,
#                 "Inputs" : self.U,
#                 "Costs" : self.TotalCost,
#                 "Angle Bounds" : [
#                         min(self.X[0,:]),
#                         max(self.X[0,:])
#                     ],
#                 "Angular Velocity Bounds" : [
#                         min(self.X[1,:]),
#                         max(self.X[1,:])
#                     ],
#                 "Input 1 Bounds" : [
#                         min(self.U[0,:]),
#                         max(self.U[0,:])
#                     ],
#                 "Input 2 Bounds" : [
#                         min(self.U[1,:]),
#                         max(self.U[1,:])
#                     ],
#                 "V" : V,
#                 "Vx" : Vx,
#                 "Vxx" : Vxx,
#                 "Qu" : Qu,
#                 "Qx" : Qx,
#                 "Qux" : Qux,
#                 "Qxu" : Qxu,
#                 "Quu" : Quu,
#                 "Quu_inv" : Quu_inv,
#                 "Qxx" : Qxx
#             }
#         )
#
# def cart_dbl_pole_DDP(X_o,**params):
#     """
#     Takes in the initial position (X_o) of the system along with an optional set of parameters (params), and runs DDP to meet some desired state.
#
#     ####################################
#     ############## params ##############
#     ####################################
#
#     1) Horizon - Number of timesteps into the future we wish to program. Must be an integer. Default is 300. (NOTE: Simulation ends at t = Horizon*dt)
#
#     2) NumberOfIterations - Number of times to iterate the DDP. Must be an integer. Default is 100.
#
#     3) dt - Discrete timestep. Must be either an int, float, float32, float64, or numpy.float. Default is 0.01. (NOTE: Simulation ends at t = Horizon*dt)
#
#     4) U_o - Initial input to the system. Must be either None (meaning zero initial input to the system) or an array with shape (Horizon-1,). Default is None.
#
#     5) p_target - Target state for the system to reach. Must be a (6,1) numpy matrix. Default is numpy.matrix([[0,0,0,0,0,0]]).T.
#
#     6) LearningRate - rate at which the system converges to the new input. Must be either an int, float, float32, float64, or numpy.float and must be between 0 and 1. Default is 0.2.
#
#     7) Q_f - Terminal cost matrix. Must be a (6,6) numpy matrix. Default is 50*numpy.matrix(numpy.eye(6)). Each element should be positive.
#
#     8) R - Running cost scalar (only one input). Must be either an int, float, float32, float64, or numpy.float. Default is 0.001.
#
#     9) PlotResults - Boolean to determine if the results of the program will be plotted. Default is False.
#
#     10) AnimateResults - Boolean to determine if the results of the program will be animated. Default is False.
#
#     11) ReturnAllResults - Boolean to determine if all results should be returned. Default is False. (NOTE: If False, the system will only return the values for the last iteration (X,U).)
#
#     12) thresh - The cost threshold when the programmer will stop looking for a better solution . Must be a positive number (Default is 5).
#
#     """
#     #------------------------------------------------>
#     #----------> Possible Parameter Values ---------->
#     #------------------------------------------------>
#
#     # Horizon - Number of timesteps into the future we wish to program
#     Horizon = params.get("Horizon",300)
#     assert type(Horizon)==int,\
#         "Horizon must be an int, not "+str(type(Horizon))+". Default is 300."
#
#     # NumberOfIterations - Number of times to iterate the DDP
#     NumberOfIterations = params.get("NumberOfIterations",100)
#     assert type(NumberOfIterations)==int, \
#         "NumberOfIterations must be an int, not "+str(type(NumberOfIterations))+". Default is 100."
#
#     # dt - Discrete timestep
#     dt = params.get("dt",0.01)
#     assert str(type(dt)) in [
#             "<class 'int'>",
#             "<class 'float'>",
#             "<class 'float32'>",
#             "<class 'float64'>",
#             "<class 'numpy.float'>"], \
#         "dt must be an int, float, float32, float64, or numpy.float not "+str(type(dt))+". Default is 0.01."
#
#     # U_o - Initial input to the system.
#     U_o = params.get("U_o",None)
#     if U_o is None:
#         U = np.zeros((Horizon-1,))
#     else:
#         assert np.shape(U_o)==(Horizon-1,), "U_o must be of shape ("+str(Horizon-1)+",) not "+str(np.shape(U_o))+"."
#         U = U_o
#
#     # p_target - Target state for the system to reach.
#     p_target = params.get("p_target",np.matrix([[0,0,0,0,0,0]]).T)
#     assert (str(type(p_target))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
#             and np.shape(p_target)==(6,1)), \
#         "p_target must be a (6,1) numpy matrix."
#
#     # LearningRate - rate at which the system converges to the new input.
#     LearningRate = params.get("LearningRate",0.2)
#     assert (str(type(LearningRate)) in [
#             "<class 'int'>",
#             "<class 'float'>",
#             "<class 'float32'>",
#             "<class 'float64'>",
#             "<class 'numpy.float'>"]
#             and (0<LearningRate<=1)), \
#         "LearningRate must be an int, float, float32, float64, or numpy.float not "+str(type(LearningRate))+", and should be between 0 and 1. Default is 0.2."
#
#     # Q_f - Terminal cost matrix
#     Q_f = params.get("Q_f",50*np.matrix(np.eye(6)))
#     assert (str(type(Q_f))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
#             and np.shape(Q_f)==(6,6)), \
#         "Q_f must be a (6,1) numpy matrix."
#
#     # R - Running cost scalar (only one input).
#     R = params.get("R",1e-3)
#     assert (str(type(R)) in [
#             "<class 'int'>",
#             "<class 'float'>",
#             "<class 'float32'>",
#             "<class 'float64'>",
#             "<class 'numpy.float'>"]
#             and (R>0)), \
#         "R must be an int, float, float32, float64, or numpy.float not "+str(type(R))+", and should be greater than 0. Default is 0.001."
#
#     # PlotResults - Boolean to determine if the results of the program will be plotted.
#     PlotResults = params.get("PlotResults",False)
#     assert type(PlotResults)==bool, "PlotResults must be either True or False (Default)."
#
#     # AnimateResults - Boolean to determine if the results of the program will be animated.
#     AnimateResults = params.get("AnimateResults",False)
#     assert type(AnimateResults)==bool, "AnimateResults must be either True or False (Default)."
#
#     # ReturnAllResults - Boolean to determine if all of the results will be returned.
#     ReturnAllResults = params.get("ReturnAllResults",False)
#     assert type(ReturnAllResults)==bool, "ReturnAllResults must be either True or False (Default)."
#
#     # thresh - Cost Threshold (must be a positive number)
#     thresh = params.get("thresh",5)
#     assert str(type(thresh)) in [
#             "<class 'int'>",
#             "<class 'float'>",
#             "<class 'float32'>",
#             "<class 'float64'>",
#             "<class 'numpy.float'>"] and thresh > 0, \
#         "thresh must be an int, float, float32, float64, or numpy.float not "+str(type(thresh))+" and must be positive. Default is 5."
#     #------------------------------------------------>
#     #-----------> Initializing the Problem ---------->
#     #------------------------------------------------>
#
#     # TotalX and TotalU
#     TotalX = []
#     TotalU = []
#
#     # Correction array for input, not derivative of input.
#     dU = np.zeros((Horizon-1,))
#
#     # Initial trajectory:
#     assert np.shape(X_o)==(6,), "X_o must have shape (6,) not "+str(np.shape(X_o))+"."
#     X = forward_integrate_dynamics(X_o,U=U,Horizon=Horizon,dt=dt)
#
#     TotalX.append(X)
#     TotalU.append(U)
#
#     V = [None]*Horizon # Each element must be a (6,1) matrix.
#     Vx = [None]*Horizon # Each element must be a (6,1) matrix.
#     Vxx = [None]*Horizon # Each element must be a (6,6) matrix.
#
#     Qu = [None]*Horizon # Each element must be a (1,1) matrix.
#     Qx = [None]*Horizon # Each element must be a (6,1) matrix.
#     Qux = [None]*Horizon # Each element must be a (1,6) matrix.
#     Qxu = [None]*Horizon # Each element must be a (6,1) matrix.
#     Quu = [None]*Horizon # Each element must be a (1,1) matrix.
#     Quu_inv = [None]*Horizon # Each element must be a (1,1) matrix.
#     Qxx = [None]*Horizon # Each element must be a (6,6) matrix.
#
#     TotalCost = [None]*NumberOfIterations
#
#     # StartTime = time.time()
#     statusbar = dsb(0,NumberOfIterations,title="Cart - Dbl. Pole (DDP)")
#     for k in range(NumberOfIterations):
#         #------------------------------------------------>
#         #--------> Linearization of the dynamics -------->
#         #> Quadratic Approximations of the cost function >
#         #------------------------------------------------>
#
#         Phi,B = return_linearized_dynamics_matrices(X,U,dt)
#         l,lx,lu,lux,lxu,luu,lxx = return_quadratic_cost_function_expansion_variables(X,U,R,dt)
#
#         #------------------------------------------------>
#         #--------------> Find the controls -------------->
#         #------------------------------------------------>
#
#         Vxx[Horizon-1]= Q_f
#         Vx[Horizon-1] = Q_f * (np.matrix(X[:,Horizon-1]).T - p_target)
#         V[Horizon-1] = (
#             (np.matrix(X[:,Horizon-1]).T - p_target).T
#             * Q_f
#             * (np.matrix(X[:,Horizon-1]).T - p_target)
#         )
#
#         #------------------------------------------------>
#         #----> Backpropagation of the Value Function ---->
#         #------------------------------------------------>
#
#         for j in reversed(range(Horizon-1)):
#             Qx[j] = lx[j] + Phi[j].T * Vx[j+1]
#             Qu[j] = lu[j] +  B[j].T * Vx[j+1]
#             Qxu[j] = (
#                 lxu[j]
#                 + Phi[j].T * Vxx[j+1] * B[j]
#             )
#             Qux[j] = Qxu[j].T
#             Quu[j] = (
#                 luu[j]
#                 + B[j].T * Vxx[j+1] * B[j]
#             )
#             Qxx[j] = (
#                 lxx[j]
#                 + Phi[j].T * Vxx[j+1] * Phi[j]
#             )
#
#             Quu_inv[j] = Quu[j]**(-1)
#
#             Vxx[j] = Qxx[j] - Qxu[j] * Quu_inv[j] * Qux[j]
#             Vx[j]= Qx[j] - Qxu[j] * Quu_inv[j] * Qu[j]
#             # V[j] = l[j] + V[j+1] - 0.5 * Qu[j].T * Quu_inv[j] * Qu[j]
#             V[j] = V[j+1] - 0.5*Qu[j].T * Quu_inv[j] * Qu[j]
#
#         #------------------------------------------------>
#         #-------------> Find the controlls -------------->
#         #------------------------------------------------>
#
#         U_new = np.zeros((Horizon-1,))
#         dX = np.matrix(np.zeros((6,1)))
#         for i in range(Horizon-1):
#             dU = -Quu_inv[j]*(Qu[i] + Qux[i]*dX)
#             dX = Phi[i]*dX + B[i]*dU
#             U_new[i] = U[i] + LearningRate*dU
#
#         U = U_new
#
#         #------------------------------------------------>
#         #-----> Simulation of the Nonlinear System ------>
#         #------------------------------------------------>
#
#         X = forward_integrate_dynamics(X_o,U=U,Horizon=Horizon,dt=dt)
#
#         TotalX.append(X)
#         TotalU.append(U)
#         TotalCost[k] =  return_cost_for_a_given_trial(
#                 X,
#                 U,
#                 p_target,
#                 dt,
#                 Q_f,
#                 R
#             )
#
#         if TotalCost[k]<thresh:
#             statusbar.update(NumberOfIterations)
#         else:
#             statusbar.update(k)
#     #     print(
#     #         'DDP Iteration %d,  Current Cost = %f \n'
#     #         % (k+1,TotalCost[k])
#     #     )
#     #
#     # print("Total Run Time: " + '%.2f'%(time.time()-StartTime) + "s")
#     Endtime = Horizon*dt
#     Time = np.arange(0,Endtime,dt)
#
#     #------------------------------------------------------->
#     #--------------------> Plot Results -------------------->
#     #------------------------------------------------------->
#
#     if PlotResults == True:
#         plot_trajectory(Time,X,TotalCost,**params)
#
#     #------------------------------------------------------->
#     #------------------> Animate Results ------------------->
#     #------------------------------------------------------->
#
#     if AnimateResults == True:
#         animate_trajectory(Time,X,U)
#
#     #------------------------------------------------------->
#     #-------------------> Return Results ------------------->
#     #------------------------------------------------------->
#
#     if ReturnAllResults == True:
#         return(
#             {
#                 "States" : TotalX,
#                 "Inputs" : TotalU,
#                 "Costs" : TotalCost,
#                 "params" : params,
#                 "Proximal Angle Bounds" : [
#                         min(X[1,:]),
#                         max(X[1,:])
#                     ],
#                 "Distal Angle Bounds" : [
#                         min(X[2,:]),
#                         max(X[2,:])
#                     ],
#                 "X Position Bounds" : [
#                         min(X[0,:]),
#                         max(X[0,:])
#                     ],
#                 "Proximal Angular Velocity Bounds" : [
#                         min(X[4,:]),
#                         max(X[4,:])
#                     ],
#                 "Distal Angular Velocity Bounds" : [
#                         min(X[5,:]),
#                         max(X[5,:])
#                     ],
#                 "X Velocity Bounds" : [
#                         min(X[3,:]),
#                         max(X[3,:])
#                     ],
#                 "Input Bounds" : [
#                         min(U),
#                         max(U)
#                     ],
#                 "V" : V,
#                 "Vx" : Vx,
#                 "Vxx" : Vxx
#             }
#         )
#     else:
#         return(TotalX[-1],TotalU[-1])

# X_o = np.array([0,-np.pi,-np.pi,0,0,0])
#
# DDP = Cart_Dbl_Pole_DDP(X_o,**params)
#
# DDP.run_ddp()
#
# DDP.animate_trajectory(
#         SaveAsGif=False,
#         FileName="Cart_Dbl_Pole_DDP_100Hz"
#     )
# DDP.plot_trajectory()
