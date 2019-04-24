#######################################################
#######################################################
#########  Differential Dynamic Programming  ##########
#######   for Cart-Pole Rigid Body Dynamics    ########
#######################################################
#######################################################
##                                                   ##
## Course: DDP Derivation and Control System Studies ##
##           Author: Evangelos, Theodorou            ##
##        Adapted for PYTHON by: Hagen, Daniel       ##
##                                                   ##
#######################################################
#######################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.patches as patches
from params import *
from quadratic_cost_function_expansion_variables import *
from quadratic_dynamics_expansion import *
from linearized_dynamics import *
from feedforward_simulation import *
from cost_function import *
from animate import *
from plot import *
from danpy.sb import dsb
import time

class Cart_Pole_DDP:
    def __init__(self,X_o,**params):
        """
        Takes in the initial position (X_o) of the system along with an optional set of parameters (params), and runs DDP to meet some desired state.

        ####################################
        ############## params ##############
        ####################################

        1) Horizon - Number of timesteps into the future we wish to program. Must be an integer. Default is 300. (NOTE: Simulation ends at t = Horizon*dt)

        2) NumberOfIterations - Number of times to iterate the DDP. Must be an integer. Default is 100.

        3) dt - Discrete timestep. Must be either an int, float, float32, float64, or numpy.float. Default is 0.01. (NOTE: Simulation ends at t = Horizon*dt)

        4) U_o - Initial input to the system. Must be either None (meaning zero initial input to the system) or an array with shape (Horizon-1,). Default is None.

        5) p_target - Target state for the system to reach. Must be a (4,1) numpy matrix. Default is numpy.matrix([[0,0,0,0]]).T.

        6) LearningRate - rate at which the system converges to the new input. Must be either an int, float, float32, float64, or numpy.float and must be between 0 and 1. Default is 0.2.

        7) Q_f - Terminal cost matrix. Must be a (4,4) numpy matrix. Default is 50*numpy.matrix(numpy.eye(4)). Each element should be positive.

        8) R - Running cost scalar (only one input). Must be either an int, float, float32, float64, or numpy.float. Default is 0.001.

        9) PlotResults - Boolean to determine if the results of the program will be plotted. Default is False.

        10) AnimateResults - Boolean to determine if the results of the program will be animated. Default is False.

        11) ReturnAllResults - Boolean to determine if all results should be returned. Default is False. (NOTE: If False, the system will only return the values for the last iteration (X,U).)


        """
        #------------------------------------------------>
        #----------> Possible Parameter Values ---------->
        #------------------------------------------------>

        # Horizon - Number of timesteps into the future we wish to program
        self.Horizon = params.get("Horizon",300)
        assert type(self.Horizon)==int,\
            "Horizon must be an int, not "+str(type(self.Horizon))+". Default is 300."

        # NumberOfIterations - Number of times to iterate the DDP
        self.NumberOfIterations = params.get("NumberOfIterations",100)
        assert type(self.NumberOfIterations)==int, \
            "NumberOfIterations must be an int, not "+str(type(self.NumberOfIterations))+". Default is 100."

        # dt - Discrete timestep
        self.dt = params.get("dt",0.01)
        assert str(type(self.dt)) in [
                "<class 'int'>",
                "<class 'float'>",
                "<class 'float32'>",
                "<class 'float64'>",
                "<class 'numpy.float'>"], \
            "dt must be an int, float, float32, float64, or numpy.float not "+str(type(self.dt))+". Default is 0.01."

        # U_o - Initial input to the system.
        self.U_o = params.get("U_o",None)
        if self.U_o is None:
            self.U = np.zeros((self.Horizon-1,))
        else:
            assert np.shape(self.U_o)==(self.Horizon-1,), "U_o must be of shape ("+str(self.Horizon-1)+",) not "+str(np.shape(self.U_o))+"."
            self.U = self.U_o

        # p_target - Target state for the system to reach.
        self.p_target = params.get("p_target",np.matrix([[0,0,0,0]]).T)
        assert (str(type(self.p_target))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
                and np.shape(self.p_target)==(4,1)), \
            "p_target must be a (4,1) numpy matrix."

        # LearningRate - rate at which the system converges to the new input.
        self.LearningRate = params.get("LearningRate",0.2)
        assert (str(type(self.LearningRate)) in [
                "<class 'int'>",
                "<class 'float'>",
                "<class 'float32'>",
                "<class 'float64'>",
                "<class 'numpy.float'>"]
                and (0<self.LearningRate<=1)), \
            "LearningRate must be an int, float, float32, float64, or numpy.float not "+str(type(self.LearningRate))+", and should be between 0 and 1. Default is 0.2."

        # Q_f - Terminal cost matrix
        self.Q_f = params.get("Q_f",50*np.matrix(np.eye(4)))
        assert (str(type(self.Q_f))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
                and np.shape(self.Q_f)==(4,4)), \
            "Q_f must be a (4,1) numpy matrix."

        # R - Running cost scalar (only one input).
        self.R = params.get("R",1e-3)
        assert (str(type(self.R)) in [
                "<class 'int'>",
                "<class 'float'>",
                "<class 'float32'>",
                "<class 'float64'>",
                "<class 'numpy.float'>"]
                and (self.R>0)), \
            "R must be an int, float, float32, float64, or numpy.float not "+str(type(self.R))+", and should be greater than 0. Default is 0.001."

        self.X_o = X_o
        assert np.shape(self.X_o)==(4,), "X_o must have shape (4,) not "+str(np.shape(self.X_o))+"."
        #
        # # PlotResults - Boolean to determine if the results of the program will be plotted.
        # PlotResults = params.get("PlotResults",False)
        # assert type(PlotResults)==bool, "PlotResults must be either True or False (Default)."
        #
        # # AnimateResults - Boolean to determine if the results of the program will be animated.
        # AnimateResults = params.get("AnimateResults",False)
        # assert type(AnimateResults)==bool, "AnimateResults must be either True or False (Default)."
        #
        # # ReturnAllResults - Boolean to determine if all of the results will be returned.
        # ReturnAllResults = params.get("ReturnAllResults",False)
        # assert type(ReturnAllResults)==bool, "ReturnAllResults must be either True or False (Default)."

    def set_Horizon(self,Horizon):
        """
        Horizon - Number of timesteps into the future we wish to program. Must be an integer. Default is 300. (NOTE: Simulation ends at t = Horizon*dt)
        """
        self.Horizon = Horizon
        assert type(self.Horizon)==int,\
            "Horizon must be an int, not "+str(type(self.Horizon))+". Default is 300."
    def set_NumberOfIterations(self,NumberOfIterations):
        """
        NumberOfIterations - Number of times to iterate the DDP. Must be an integer. Default is 100.
        """
        self.NumberOfIterations = NumberOfIterations
        assert type(self.NumberOfIterations)==int, \
            "NumberOfIterations must be an int, not "+str(type(self.NumberOfIterations))+". Default is 100."
    def set_dt(self,dt):
        """
        dt - Discrete timestep. Must be either an int, float, float32, float64, or numpy.float. Default is 0.01. (NOTE: Simulation ends at t = Horizon*dt)
        """
        self.dt = dt
        assert str(type(self.dt)) in [
                "<class 'int'>",
                "<class 'float'>",
                "<class 'float32'>",
                "<class 'float64'>",
                "<class 'numpy.float'>"], \
            "dt must be an int, float, float32, float64, or numpy.float not "+str(type(self.dt))+". Default is 0.01."
    def set_U_o(self,U_o):
        """
        U_o - Initial input to the system. Must be either None (meaning zero initial input to the system) or an array with shape (Horizon-1,). Default is None.
        """
        self.U_o = U_o
        if self.U_o is None:
            self.U = np.zeros((self.Horizon-1,))
        else:
            assert np.shape(self.U_o)==(self.Horizon-1,), "U_o must be of shape ("+str(self.Horizon-1)+",) not "+str(np.shape(self.U_o))+"."
            self.U = self.U_o
    def set_X_o(self,X_o):
        """
        X_o - Initial states of the system. Must be of shape (4,).
        """
        self.X_o = X_o
        assert np.shape(self.X_o)==(4,), "X_o must have shape (4,) not "+str(np.shape(self.X_o))+"."
    def set_p_target(self,p_target):
        """
        p_target - Target state for the system to reach. Must be a (4,1) numpy matrix. Default is numpy.matrix([[0,0,0,0]]).T.
        """
        self.p_target = p_target
        assert (str(type(self.p_target))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
                and np.shape(self.p_target)==(4,1)), \
            "p_target must be a (4,1) numpy matrix."
    def set_LearningRate(self,LearningRate):
        """
        LearningRate - rate at which the system converges to the new input. Must be either an int, float, float32, float64, or numpy.float and must be between 0 and 1. Default is 0.2.
        """
        self.LearningRate = LearningRate
        assert (str(type(self.LearningRate)) in [
                "<class 'int'>",
                "<class 'float'>",
                "<class 'float32'>",
                "<class 'float64'>",
                "<class 'numpy.float'>"]
                and (0<self.LearningRate<=1)), \
            "LearningRate must be an int, float, float32, float64, or numpy.float not "+str(type(self.LearningRate))+", and should be between 0 and 1. Default is 0.2."
    def set_Q_f(self,Q_f):
        """
        Q_f - Terminal cost matrix. Must be a (4,4) numpy matrix. Default is 50*numpy.matrix(numpy.eye(4)). Each element should be positive.
        """
        self.Q_f = Q_f
        assert (str(type(self.Q_f))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
                and np.shape(self.Q_f)==(4,4)), \
            "Q_f must be a (4,1) numpy matrix."
    def set_R(self,R):
        """
        R - Running cost scalar (only one input). Must be either an int, float, float32, float64, or numpy.float. Default is 0.001.
        """
        self.R = R
        assert (str(type(self.R)) in [
                "<class 'int'>",
                "<class 'float'>",
                "<class 'float32'>",
                "<class 'float64'>",
                "<class 'numpy.float'>"]
                and (self.R>0)), \
            "R must be an int, float, float32, float64, or numpy.float not "+str(type(self.R))+", and should be greater than 0. Default is 0.001."

    def forward_integrate_dynamics(self):
        """
        ICs must be a list of floats and/or ints of length 4. If ReturnX is True, the this will return an array of shape (4,len(Time)).

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        **kwargs
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        UsingDegrees must be a bool. Default is True. If True, then the ICs for pendulum angle and angular velocity can be given in degrees and degrees per second, respectively.

        AnimateStates must be a bool. Default is False. If True, the program will run animate_trajectory().

        PlotStates must be a bool. Default is False. If True, the program will run plot the resulting states.

        dt must be a number. Default is 0.01. Used with Horizon to define the time array (Time).

        Horizon must be a number. Default is 300. Used with dt to define the time array (Time).

        U can either be None (default) or can be an array with lenth (len(Time)-1). If None, then U will be chosen to be np.zeros(len(Time)-1)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Notes:

        X1: cart X-position
        X2: angle of pendulum
        X3: cart X-velocity
        X4: angular velocity of pendulum

        """

        self.X[0,0] = self.X_o[0]
        self.X[1,0] = self.X_o[1]
        self.X[2,0] = self.X_o[2]
        self.X[3,0] = self.X_o[3]

        for i in range(self.Horizon-1):
            self.X[0,i+1] = self.X[0,i] + F1(self.X[:,i],self.U[i])*self.dt
            self.X[1,i+1] = self.X[1,i] + F2(self.X[:,i],self.U[i])*self.dt
            self.X[2,i+1] = self.X[2,i] + F3(self.X[:,i],self.U[i])*self.dt
            self.X[3,i+1] = self.X[3,i] + F4(self.X[:,i],self.U[i])*self.dt

    def return_Phi(self,x,u):
        """
        Takes in the state vector (X), the input vector (U) and returns the discretized and linearized state matrix, Phi.

        NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).

        #######################
        ##### NEED TO DO: #####
        #######################

        [ ] - Create tests that ensure that X and U are the correct dimensions.
        [ ] - Create tests to make sure that the outputs are of the correct sizes.
        """
        assert (str(type(x)) in ["<class 'numpy.ndarray'>"]
                and np.shape(x)==(4,)), "Error with the type and shape of x ["+ return_Phi.__name__+"()]."
        assert str(type(u)) in ["<class 'int'>",
                "<class 'float'>",
                "<class 'numpy.float'>",
                "<class 'numpy.float64'>",
                "<class 'numpy.int32'>",
                "<class 'numpy.int64'>"],\
            "u must be a number. Not " + str(type(u)) + "."

        # Removed the u split into two scalars because u is already a scalar.

        h1 = np.array([h,0,0,0])
        h2 = np.array([0,h,0,0])
        h3 = np.array([0,0,h,0])
        h4 = np.array([0,0,0,h])

        # Build the dFx matrix

        dFx = np.zeros((4,4))

        # dFx[0,0] = 0 # dF1/dx1⋅dx1 = (self.F1(x,u)-self.F1(x-h1,u))/h = 0
        # dFx[0,1] = 0 # dF1/dx2⋅dx2 = (self.F1(x,u)-self.F1(x-h2,u))/h = 0
        dFx[0,2] = 1 # dF1/dx3⋅dx3 = (self.F1(x,u)-self.F1(x-h3,u))/h = 1
        # dFx[0,3] = 0 # dF1/dx4⋅dx4 = (self.F1(x,u)-self.F1(x-h4,u))/h = 0

        # dFx[1,0] = 0 # dF2/dx1⋅dx2 = (self.F2(x,u)-self.F2(x-h1,u))/h = 0
        # dFx[1,1] = 0 # dF2/dx2⋅dx2 = (self.F2(x,u)-self.F2(x-h2,u))/h = 0
        # dFx[1,2] = 0 # dF2/dx3⋅dx2 = (self.F2(x,u)-self.F2(x-h3,u))/h = 1
        dFx[1,3] = 1 # dF2/dx4⋅dx2 = (self.F2(x,u)-self.F2(x-h4,u))/h = 0

        # F3 is the acceleration of the cart.
        dFx[2,0] = (F3(x,u)-F3(x-h1,u))/h
        dFx[2,1] = (F3(x,u)-F3(x-h2,u))/h
        dFx[2,2] = (F3(x,u)-F3(x-h3,u))/h
        dFx[2,3] = (F3(x,u)-F3(x-h4,u))/h

        # F4 is the angular acceleration of the pendulum.
        dFx[3,0] = (F4(x,u)-F4(x-h1,u))/h
        dFx[3,1] = (F4(x,u)-F4(x-h2,u))/h
        dFx[3,2] = (F4(x,u)-F4(x-h3,u))/h
        dFx[3,3] = (F4(x,u)-F4(x-h4,u))/h

        Phi = np.matrix(np.eye(4) + dFx*self.dt)
        assert np.shape(Phi)==(4,4) \
            and str(type(Phi))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
        "Phi must be a (4,4) numpy matrix. Not " + str(type(Phi)) + " of shape " + str(np.shape(Phi)) + "."
        return(Phi)
    def return_B(self,x,u):
        """
        Takes in the state vector (x), the input vector (u) and returns the discretized and linearized input matrix, B.

        NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).

        #######################
        ##### NEED TO DO: #####
        #######################

        [ ] - Create tests that ensure that x and u are the correct dimensions.
        [ ] - Create tests to make sure that the outputs are of the correct sizes.
        """
        assert (str(type(x)) in ["<class 'numpy.ndarray'>"]
                and np.shape(x)==(4,)), "Error with the type and shape of x ["+ return_B.__name__+"()]."
        assert str(type(u)) in ["<class 'int'>",
                "<class 'float'>",
                "<class 'numpy.float'>",
                "<class 'numpy.float64'>",
                "<class 'numpy.int32'>",
                "<class 'numpy.int64'>"],\
            "u must be a number. Not " + str(type(u)) + "."

        # Removed the u split into two scalars because u is already a scalar.

        h1 = np.array([h,0,0,0])
        h2 = np.array([0,h,0,0])
        h3 = np.array([0,0,h,0])
        h4 = np.array([0,0,0,h])

        #Build the dFu matrix

        dFu = np.zeros((4,1))

        dFu[0,0] = 0
        dFu[1,0] = 0
        dFu[2,0] = (F3(x,u)-F3(x,u-h))/h
        dFu[3,0] = (F4(x,u)-F4(x,u-h))/h

        B = np.matrix(dFu*self.dt)
        assert np.shape(B)==(4,1) \
                and str(type(B))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
            "B must be a (4,1) numpy matrix. Not " + str(type(B)) + " of shape " + str(np.shape(B)) + "."

        return(B)
    def return_linearized_dynamics_matrices(self):
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
                    lambda x,u: self.return_Phi(x,u),
                    self.X[:,:-1].T,
                    self.U
                )
            )

        B = list(
                map(
                    lambda x,u: self.return_B(x,u),
                    self.X[:,:-1].T,
                    self.U
                )
            )
        return(Phi,B)

    def return_quadratic_cost_function_expansion_variables(self):
        """
        Takes in the input U and the the corresponding output X, as well as dt and returns lists that contain the coefficient matrices for the quadratic expansion of the cost function (l(x,u)) for each timestep for range(len(Time)-1).
        """
        # returns a list of length len(Time)-1, each element with shape (1,1), where n is the number of states.
        l = list(
                map(
                    lambda x,u: u.T * self.R * u * self.dt,
                    self.X[:,1:].T,
                    self.U
                )
            )

        # returns a list of length len(Time)-1, each element with shape (n,1), where n is the number of states.
        lx = list(
                map(
                    lambda x,u: np.matrix(np.zeros((4,1)))*self.dt,
                    self.X[:,1:].T,
                    self.U
                )
            )

        # returns a list of length len(Time)-1, each element with shape (m,1), where n is the number of states.
        lu = list(
                map(
                    lambda x,u: self.R * u * self.dt,
                    self.X[:,1:].T,
                    self.U
                )
            )

        # returns a list of length len(Time)-1, each element with shape (m,n), where m is the number of inputs and n is the number of states.
        lux = list(
                map(
                    lambda x,u: np.matrix(np.zeros((1,4)))*self.dt,
                    self.X[:,1:].T,
                    self.U
                )
            )

        # returns a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
        lxu = list(
                map(
                    lambda x,u: np.matrix(np.zeros((4,1)))*self.dt,
                    self.X[:,1:].T,
                    self.U
                )
            )

        # returns a list of length len(Time)-1, each element with shape (m,m), where m is the number of inputs.
        luu = list(
                map(
                    lambda x,u: self.R*self.dt,
                    self.X[:,1:].T,
                    self.U
                )
            )

        # returns a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
        lxx = list(
                map(
                    lambda x,u: np.matrix(np.zeros((4,4)))*self.dt,
                    self.X[:,1:].T,
                    self.U
                )
            )

        return(l,lx,lu,lux,lxu,luu,lxx)

    def return_cost_for_a_given_trial(self):
        """
        This takes in the state variables over time (X) and the input (U), as well as the target state (p_target), the time step (dt), and the cost matrices (Q_f and R), and output the cost of the trial.
        """

        RunningCost = 0
        for j in range(self.Horizon-1):
            RunningCost = RunningCost + 0.5 * self.U[j].T * self.R * self.U[j] * self.dt

        TerminalCost = (
            (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target).T
            * self.Q_f
            * (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target)
        )[0,0]

        Cost = RunningCost + TerminalCost
        return(Cost)

    def run_ddp(self):
        #------------------------------------------------>
        #-----------> Initializing the Problem ---------->
        #------------------------------------------------>

        # TotalX and TotalU
        TotalX = []
        TotalU = []

        # Correction array for input, not derivative of input.
        dU = np.zeros((self.Horizon-1,))

        # Initial trajectory:
        assert np.shape(self.X_o)==(4,), "X_o must have shape (4,) not "+str(np.shape(self.X_o))+"."
        self.X = np.zeros((4,self.Horizon))
        self.forward_integrate_dynamics()

        V = [None]*self.Horizon # Each element must be a (4,1) matrix.
        Vx = [None]*self.Horizon # Each element must be a (4,1) matrix.
        Vxx = [None]*self.Horizon # Each element must be a (4,4) matrix.

        Qu = [None]*self.Horizon # Each element must be a (1,1) matrix.
        Qx = [None]*self.Horizon # Each element must be a (4,1) matrix.
        Qux = [None]*self.Horizon # Each element must be a (1,4) matrix.
        Qxu = [None]*self.Horizon # Each element must be a (4,1) matrix.
        Quu = [None]*self.Horizon # Each element must be a (1,1) matrix.
        Quu_inv = [None]*self.Horizon # Each element must be a (1,1) matrix.
        Qxx = [None]*self.Horizon # Each element must be a (4,4) matrix.

        self.TotalCost = [None]*self.NumberOfIterations

        StartTime = time.time()
        for k in range(self.NumberOfIterations):
            #------------------------------------------------>
            #--------> Linearization of the dynamics -------->
            #> Quadratic Approximations of the cost function >
            #------------------------------------------------>

            Phi,B = self.return_linearized_dynamics_matrices()
            l,lx,lu,lux,lxu,luu,lxx = self.return_quadratic_cost_function_expansion_variables()

            #------------------------------------------------>
            #--------------> Find the controls -------------->
            #------------------------------------------------>

            Vxx[self.Horizon-1]= self.Q_f
            Vx[self.Horizon-1] = self.Q_f * (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target)
            V[self.Horizon-1] = (
                (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target).T
                * self.Q_f
                * (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target)
            )

            #------------------------------------------------>
            #----> Backpropagation of the Value Function ---->
            #------------------------------------------------>

            for j in reversed(range(self.Horizon-1)):
                Qx[j] = lx[j] + Phi[j].T * Vx[j+1]
                Qu[j] = lu[j] +  B[j].T * Vx[j+1]
                Qxu[j] = (
                    lxu[j]
                    + Phi[j].T * Vxx[j+1] * B[j]
                )
                Qux[j] = Qxu[j].T
                Quu[j] = (
                    luu[j]
                    + B[j].T * Vxx[j+1] * B[j]
                )
                Qxx[j] = (
                    lxx[j]
                    + Phi[j].T * Vxx[j+1] * Phi[j]
                )

                Quu_inv[j] = Quu[j]**(-1)

                Vxx[j] = Qxx[j] - Qxu[j] * Quu_inv[j] * Qux[j]
                Vx[j]= Qx[j] - Qxu[j] * Quu_inv[j] * Qu[j]
                # V[j] = l[j] + V[j+1] - 0.5 * Qu[j].T * Quu_inv[j] * Qu[j]
                V[j] = V[j+1] - 0.5*Qu[j].T * Quu_inv[j] * Qu[j]

            #------------------------------------------------>
            #-------------> Find the controlls -------------->
            #------------------------------------------------>

            U_new = np.zeros((self.Horizon-1,))
            dX = np.matrix(np.zeros((4,1)))
            for i in range(self.Horizon-1):
                dU = -Quu_inv[i]*(Qu[i] + Qux[i]*dX)
                dX = Phi[i]*dX + B[i]*dU
                U_new[i] = self.U[i] + self.LearningRate*dU

            self.U = U_new

            #------------------------------------------------>
            #-----> Simulation of the Nonlinear System ------>
            #------------------------------------------------>

            self.forward_integrate_dynamics()

            self.TotalCost[k] =  self.return_cost_for_a_given_trial()

            # print(
            #     'DDP Iteration %d,  Current Cost = %f \n'
            #     % (k+1,self.TotalCost[k])
            # )
    def run_ddp_for_mpc(self):
        #------------------------------------------------>
        #-----------> Initializing the Problem ---------->
        #------------------------------------------------>

        # TotalX and TotalU
        TotalX = []
        TotalU = []

        # Correction array for input, not derivative of input.
        dU = np.zeros((self.Horizon-1,))

        # Initial trajectory:
        assert np.shape(self.X_o)==(4,), "X_o must have shape (4,) not "+str(np.shape(self.X_o))+"."
        self.X = np.zeros((4,self.Horizon))
        self.forward_integrate_dynamics()

        V = [None]*self.Horizon # Each element must be a (4,1) matrix.
        Vx = [None]*self.Horizon # Each element must be a (4,1) matrix.
        Vxx = [None]*self.Horizon # Each element must be a (4,4) matrix.

        Qu = [None]*self.Horizon # Each element must be a (1,1) matrix.
        Qx = [None]*self.Horizon # Each element must be a (4,1) matrix.
        Qux = [None]*self.Horizon # Each element must be a (1,4) matrix.
        Qxu = [None]*self.Horizon # Each element must be a (4,1) matrix.
        Quu = [None]*self.Horizon # Each element must be a (1,1) matrix.
        Quu_inv = [None]*self.Horizon # Each element must be a (1,1) matrix.
        Qxx = [None]*self.Horizon # Each element must be a (4,4) matrix.

        self.TotalCost = [None]*self.NumberOfIterations

        StartTime = time.time()
        for k in range(self.NumberOfIterations):
            #------------------------------------------------>
            #--------> Linearization of the dynamics -------->
            #> Quadratic Approximations of the cost function >
            #------------------------------------------------>

            Phi,B = self.return_linearized_dynamics_matrices()
            l,lx,lu,lux,lxu,luu,lxx = self.return_quadratic_cost_function_expansion_variables()

            #------------------------------------------------>
            #--------------> Find the controls -------------->
            #------------------------------------------------>

            Vxx[self.Horizon-1]= self.Q_f
            Vx[self.Horizon-1] = self.Q_f * (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target)
            V[self.Horizon-1] = (
                (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target).T
                * self.Q_f
                * (np.matrix(self.X[:,self.Horizon-1]).T - self.p_target)
            )

            #------------------------------------------------>
            #----> Backpropagation of the Value Function ---->
            #------------------------------------------------>

            for j in reversed(range(self.Horizon-1)):
                Qx[j] = lx[j] + Phi[j].T * Vx[j+1]
                Qu[j] = lu[j] +  B[j].T * Vx[j+1]
                Qxu[j] = (
                    lxu[j]
                    + Phi[j].T * Vxx[j+1] * B[j]
                )
                Qux[j] = Qxu[j].T
                Quu[j] = (
                    luu[j]
                    + B[j].T * Vxx[j+1] * B[j]
                )
                Qxx[j] = (
                    lxx[j]
                    + Phi[j].T * Vxx[j+1] * Phi[j]
                )

                Quu_inv[j] = Quu[j]**(-1)

                Vxx[j] = Qxx[j] - Qxu[j] * Quu_inv[j] * Qux[j]
                Vx[j]= Qx[j] - Qxu[j] * Quu_inv[j] * Qu[j]
                # V[j] = l[j] + V[j+1] - 0.5 * Qu[j].T * Quu_inv[j] * Qu[j]
                V[j] = V[j+1] - 0.5*Qu[j].T * Quu_inv[j] * Qu[j]

            #------------------------------------------------>
            #-------------> Find the controlls -------------->
            #------------------------------------------------>
            if k==self.NumberOfIterations-1:
                dX = np.matrix(np.zeros((4,1)))
                dU = -Quu_inv[0]*(Qu[0] + Qux[0]*dX)
                self.U = (self.U[0] + self.LearningRate*dU)*np.ones((self.Horizon-1,))
                self.next_X = np.zeros((4,1))
                self.next_X[0,0] = self.X_o[0] + F1(self.X_o,self.U[0])*self.dt
                self.next_X[1,0] = self.X_o[1] + F2(self.X_o,self.U[0])*self.dt
                self.next_X[2,0] = self.X_o[2] + F3(self.X_o,self.U[0])*self.dt
                self.next_X[3,0] = self.X_o[3] + F4(self.X_o,self.U[0])*self.dt
            else:
                U_new = np.zeros((self.Horizon-1,))
                dX = np.matrix(np.zeros((4,1)))
                for i in range(self.Horizon-1):
                    dU = -Quu_inv[i]*(Qu[i] + Qux[i]*dX)
                    dX = Phi[i]*dX + B[i]*dU
                    U_new[i] = self.U[i] + self.LearningRate*dU

                self.U = U_new

                #------------------------------------------------>
                #-----> Simulation of the Nonlinear System ------>
                #------------------------------------------------>

                self.forward_integrate_dynamics()

                self.TotalCost[k] =  self.return_cost_for_a_given_trial()

                # print(
                #     'DDP Iteration %d,  Current Cost = %f \n'
                #     % (k+1,self.TotalCost[k])
                # )

    def return_time_array(self):
        Endtime = self.Horizon*self.dt
        Time = np.arange(0,Endtime,self.dt)
        return(Time)
    def plot_trajectory(self):
        Time = self.return_time_array()

        fig1 = plt.figure(figsize=(12,8))
        plt.suptitle("Cart Pole Control via DDP",fontsize=16)

        plt.subplot(321)
        plt.plot(
            Time,
            self.p_target[0,0]*np.ones((self.Horizon,)),
            'r--',
            linewidth=2
        )
        plt.plot(Time,self.X[0,:],linewidth=4)
        # plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('X Position',fontsize=16)
        plt.grid(True)

        plt.subplot(322)
        plt.plot(
            Time,
            self.p_target[1,0]*np.ones((self.Horizon,)),
            'r--',
            linewidth=2
        )
        plt.plot(Time,self.X[1,:],linewidth=4)
        # plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('Theta',fontsize=16)
        plt.grid(True)

        plt.subplot(323)
        plt.plot(
            Time,
            self.p_target[2,0]*np.ones((self.Horizon,)),
            'r--',
            linewidth=4
        )
        plt.plot(Time,self.X[2,:],linewidth=4)
        # plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('X velocity',fontsize=16)
        plt.grid(True)

        plt.subplot(324)
        plt.plot(
            Time,
            self.p_target[3,0]*np.ones((self.Horizon,)),
            'r--',
            linewidth=4
        )
        plt.plot(Time,self.X[3,:],linewidth=4)
        # plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('Angular Velocity',fontsize=16)
        plt.grid(True)

        plt.subplot(3,2,(5,6))
        plt.plot(self.TotalCost,linewidth=2)
        plt.xlabel('Iterations',fontsize=16)
        plt.ylabel('Cost',fontsize=16)
        plt.show()
    def animate_trajectory(self,**kwargs):
        Time = self.return_time_array()

        SaveAsGif = kwargs.get("SaveAsGif",False)
        assert type(SaveAsGif)==bool, "SaveAsGif must be either True or False (Default)."

        FileName = kwargs.get("FileName","ddp_cart_pole")
        assert type(FileName)==str,"FileName must be a str."

            # Angles must be in degrees for animation

        X2d = self.X[1,:]*(180/np.pi)
        X4d = self.X[3,:]*(180/np.pi)


        fig = plt.figure(figsize=(12,10))
        ax0 = plt.subplot2grid((3,4),(0,0),colspan=2) # animation
        ax1 = plt.subplot2grid((3,4),(0,2),colspan=2) # input
        ax2 = plt.subplot2grid((3,4),(1,0),colspan=2) # pendulum angle
        ax3 = plt.subplot2grid((3,4),(1,2),colspan=2) # cart position
        ax4 = plt.subplot2grid((3,4),(2,0),colspan=2) # pendulum angular velocity
        ax5 = plt.subplot2grid((3,4),(2,2),colspan=2) # cart velocty

        plt.suptitle("Cart-Pendulum Example",Fontsize=28,y=0.95)

        Pendulum_Width = 0.01*L
        Pendulum_Length = 2*L
        Cart_Width = 4*L
        Cart_Height = 2*L
        Wheel_Radius = 0.125*Cart_Width
        # Model Drawing
        marker_interdistance = 25
        lowest_marker = marker_interdistance* \
                (int(np.floor(self.X[0].min())/marker_interdistance)-1)
        highest_marker = marker_interdistance* \
                (int(np.ceil(self.X[0].max())/marker_interdistance)+2)
        markers = np.arange(
                lowest_marker,
                highest_marker,
                marker_interdistance
                )
        smallmarkers = np.arange(
                lowest_marker,
                highest_marker,
                marker_interdistance/2
                )
        marker_str = []
        for marker in markers:
            if marker%100==0:
                marker_str.append(str(int(marker)))
            else:
                marker_str.append("")

        Markers = ax0.scatter(
                    markers-self.X[0,0],
                    -0.90*(Cart_Height/2 + Wheel_Radius*2)*np.ones(len(markers)),
                    marker="|",
                    c="k",
                    s=2000)

        SmallMarkers = ax0.scatter(
                    smallmarkers-self.X[0,0],
                    -0.90*(Cart_Height/2
                        + Wheel_Radius*2)*np.ones(len(smallmarkers)),
                    marker="|",
                    c="k",
                    s=1000)

        Cart = plt.Rectangle(
                    (self.X[0,0]-Cart_Width/2,-Cart_Height/2),
                    Cart_Width,
                    Cart_Height,
                    Color='#4682b4')
        ax0.add_patch(Cart)

        FrontWheel = plt.Circle(
                    (self.X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    Wheel_Radius,
                    Color='k')
        ax0.add_patch(FrontWheel)
        FrontWheel_Rivet = plt.Circle(
                    (self.X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    0.2*Wheel_Radius,
                    Color='0.70')
        ax0.add_patch(FrontWheel_Rivet)

        BackWheel = plt.Circle(
                    (self.X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    Wheel_Radius,
                    Color='k')
        ax0.add_patch(BackWheel)
        BackWheel_Rivet = plt.Circle(
                    (self.X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    0.2*Wheel_Radius,
                    Color='0.70')
        ax0.add_patch(BackWheel_Rivet)

        Pendulum, = ax0.plot(
                        [
                            self.X[0,0],
                            self.X[0,0] + Pendulum_Length*np.sin(self.X[1,0])
                        ],
                        [
                            Cart_Height/2 + Pendulum_Width/2,
                            Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(self.X[1,0])
                        ],
                        Color='0.50',
                        lw = 5,
                        solid_capstyle='round'
                        )


        Pendulum_Attachment = plt.Circle((self.X[0,0],Cart_Height/2),100*Pendulum_Width/2,Color='#4682b4')
        ax0.add_patch(Pendulum_Attachment)

        Pendulum_Rivet, = ax0.plot(
            [self.X[0,0]],
            [Cart_Height/2 + Pendulum_Width/2],
            c='k',
            marker='o',
            lw=1
            )

        MinimumX = self.X[0,0]-13
        MaximumX = self.X[0,0]+13
        # if max(abs(X[1,:]-X[1,0]))<1e-7:
        #     MinimumX = X[1,0]-10
        #     MaximumX = X[1,0]+10
        # elif max(abs(X[1,:]-X[1,0]))<2:
        #     MinimumX = min(X[1,:])-1.25*Cart_Width/2-5
        #     MaximumX = max(X[1,:])+1.25*Cart_Width/2+5
        # else:
        #     MinimumX = min(X[1,:])-1.25*Cart_Width/2
        #     MaximumX = max(X[1,:])+1.25*Cart_Width/2

        Ground = plt.Rectangle(
                    (MinimumX,-1.50*(Cart_Height/2 + Wheel_Radius*2)),
                    MaximumX-MinimumX,
                    0.50*(Cart_Height/2 + Wheel_Radius*2),
                    Color='0.70')
        ax0.add_patch(Ground)

        TimeStamp = ax0.text(
            0.75*MinimumX,
            0.75*1.50*(Cart_Height/2+Pendulum_Width/2+Pendulum_Length+Pendulum_Width/2),
            "Time: "+"{:.2f}".format(Time[0])+" s",
            color='0.50',
            fontsize=16
        )

        ax0.get_xaxis().set_ticks([])
        ax0.get_yaxis().set_ticks([])
        ax0.set_frame_on(True)

        ax0.set_xlim([MinimumX,MaximumX])
        ax0.set_ylim(
            [
                -1.50*(Cart_Height/2 + Wheel_Radius*2),
                1.50*(Cart_Height/2+Pendulum_Width/2+Pendulum_Length+Pendulum_Width/2)
            ]
            )
        ax0.set_aspect('equal')

        #Input

        Input, = ax1.plot([0],[self.U[0]],color = 'b')
        ax1.set_xlim(0,Time[-1])
        ax1.set_xticks(list(np.linspace(0,Time[-1],5)))
        ax1.set_xticklabels([str(0),'','','',str(Time[-1])])
        if max(abs(self.U-self.U[0]))<1e-7:
            ax1.set_ylim([self.U[0]-2,self.U[0]+2])
        else:
            RangeU = max(self.U)-min(self.U)
            ax1.set_ylim([min(self.U)-0.1*RangeU,max(self.U)+0.1*RangeU])

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.set_title("Input Force (N)",fontsize=16,fontweight = 4,color = 'b',y = 0.95)

        #Pendulum Angle

        Angle, = ax2.plot([0],[X2d[0]],color = 'r')
        ax2.set_xlim(0,Time[-1])
        ax2.set_xticks(list(np.linspace(0,Time[-1],5)))
        ax2.set_xticklabels([str(0),'','','',str(Time[-1])])
        if max(abs(X2d-X2d[0]))<1e-7:
            ax2.set_ylim([X2d[0]-2,X2d[0]+2])
        else:
            RangeX2d= max(X2d)-min(X2d)
            ax2.set_ylim([min(X2d)-0.1*RangeX2d,max(X2d)+0.1*RangeX2d])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_title("Pendulum Angle (deg)",fontsize=16,fontweight = 4,color = 'r',y = 0.95)

        #Cart Position

        Position, = ax3.plot([0],[self.X[0,0]],color = 'g')
        ax3.set_xlim(0,Time[-1])
        ax3.set_xticks(list(np.linspace(0,Time[-1],5)))
        ax3.set_xticklabels([str(0),'','','',str(Time[-1])])
        if max(abs(self.X[0,:]-self.X[0,0]))<1e-7:
            ax3.set_ylim([self.X[0,0]-2,self.X[0,0]+2])
        else:
            RangeX1 = max(self.X[0,:])-min(self.X[0,:])
            ax3.set_ylim([min(self.X[0,:])-0.1*RangeX1,max(self.X[0,:])+0.1*RangeX1])

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.set_title("Cart Position (m)",fontsize=16,fontweight = 4,color = 'g',y = 0.95)

        # Angular Velocity

        AngularVelocity, = ax4.plot([0],[X4d[0]],color='r',linestyle='--')
        ax4.set_xlim(0,Time[-1])
        ax4.set_xticks(list(np.linspace(0,Time[-1],5)))
        ax4.set_xticklabels([str(0),'','','',str(Time[-1])])
        if max(abs(X4d-X4d[0]))<1e-7:
            ax4.set_ylim([X4d[0]-2,X4d[0]+2])
        else:
            RangeX4d= max(X4d)-min(X4d)
            ax4.set_ylim([min(X4d)-0.1*RangeX4d,max(X4d)+0.1*RangeX4d])
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.set_title("Pendulum Angular Velocity (deg/s)",fontsize=16,fontweight = 4,color = 'r',y = 0.95)

        # Cart Velocity

        Velocity, = ax5.plot([0],[self.X[2,0]],color='g',linestyle="--")
        ax5.set_xlim(0,Time[-1])
        ax5.set_xticks(list(np.linspace(0,Time[-1],5)))
        ax5.set_xticklabels([str(0),'','','',str(Time[-1])])
        if max(abs(self.X[2,:]-self.X[2,0]))<1e-7:
            ax5.set_ylim([self.X[2,0]-2,self.X[2,0]+2])
        else:
            RangeX3= max(self.X[2,:])-min(self.X[2,:])
            ax5.set_ylim([min(self.X[2,:])-0.1*RangeX3,max(self.X[2,:])+0.1*RangeX3])
        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax5.set_title("Cart Velocity (m/s)",fontsize=16,fontweight = 4,color = 'g',y = 0.95)

        def animate(i):
            offset = np.concatenate([
                    (markers-self.X[0,i])[:,np.newaxis],
                    (-0.90*(Cart_Height/2 + Wheel_Radius*2)
                        * np.ones(len(markers)))[:,np.newaxis]
                    ],
                    axis=1)
            Markers.set_offsets(offset)

            smalloffset = np.concatenate([
                    (smallmarkers-self.X[0,i])[:,np.newaxis],
                    (-0.90*(Cart_Height/2 + Wheel_Radius*2)
                        * np.ones(len(smallmarkers)))[:,np.newaxis]
                    ],
                    axis=1)
            SmallMarkers.set_offsets(smalloffset)

            Cart.xy = (self.X[0,0]-Cart_Width/2,-Cart_Height/2)

            FrontWheel.center = (self.X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
            FrontWheel_Rivet.center = (self.X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))

            BackWheel.center = (self.X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
            BackWheel_Rivet.center = (self.X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))

            Pendulum.set_xdata([self.X[0,0],self.X[0,0] + Pendulum_Length*np.sin(self.X[1,i])])
            Pendulum.set_ydata([Cart_Height/2 + Pendulum_Width/2,
                                Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(self.X[1,i])])

            Pendulum_Attachment.center = (self.X[0,0],Cart_Height/2)

            Pendulum_Rivet.set_xdata([self.X[0,0]])

            TimeStamp.set_text("Time: "+"{:.2f}".format(Time[i])+" s")

            Input.set_xdata(Time[:i])
            Input.set_ydata(self.U[:i])

            Position.set_xdata(Time[:i])
            Position.set_ydata(self.X[0,:i])

            Angle.set_xdata(Time[:i])
            Angle.set_ydata(X2d[:i])

            Velocity.set_xdata(Time[:i])
            Velocity.set_ydata(self.X[2,:i])

            AngularVelocity.set_xdata(Time[:i])
            AngularVelocity.set_ydata(X4d[:i])

            return SmallMarkers,Markers,Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Input,Position,Angle,Velocity,AngularVelocity,TimeStamp,

        # Init only required for blitting to give a clean slate.
        def init():
            Markers = plt.scatter(
                        markers-self.X[0,0],
                        -0.90*(Cart_Height/2 + Wheel_Radius*2)*np.ones(len(markers)),
                        marker="^",
                        c="k",
                        s=2000)

            SmallMarkers = plt.scatter(
                        smallmarkers-self.X[0,0],
                        -0.90*(Cart_Height/2 + Wheel_Radius*2)*np.ones(len(smallmarkers)),
                        marker="^",
                        c="k",
                        s=1000)

            Cart = plt.Rectangle(
                        (self.X[0,0]-Cart_Width/2,-Cart_Height/2),
                        Cart_Width,
                        Cart_Height,
                        Color='#4682b4')
            ax0.add_patch(Cart)

            FrontWheel = plt.Circle(
                        (self.X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                        Wheel_Radius,
                        Color='k')
            ax0.add_patch(FrontWheel)
            FrontWheel_Rivet = plt.Circle(
                        (self.X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                        0.2*Wheel_Radius,
                        Color='0.70')
            ax0.add_patch(FrontWheel_Rivet)

            BackWheel = plt.Circle(
                        (self.X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                        Wheel_Radius,
                        Color='k')
            ax0.add_patch(BackWheel)
            BackWheel_Rivet = plt.Circle(
                        (self.X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                        0.2*Wheel_Radius,
                        Color='0.70')
            ax0.add_patch(BackWheel_Rivet)

            Pendulum, = ax0.plot(
                            [
                                self.X[0,0],
                                self.X[0,0] + Pendulum_Length*np.sin(self.X[1,0])
                            ],
                            [
                                Cart_Height/2 + Pendulum_Width/2,
                                Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(self.X[1,0])
                            ],
                            Color='0.50',
                            lw = 10,
                            solid_capstyle='round'
                            )

            Pendulum_Attachment = plt.Circle((self.X[0,0],Cart_Height/2),100*Pendulum_Width/2,Color='#4682b4')
            ax0.add_patch(Pendulum_Attachment)

            Pendulum_Rivet, = ax0.plot(
                [self.X[0,0]],
                [Cart_Height/2 + Pendulum_Width/2],
                c='k',
                marker='o',
                lw=1
                )

            Ground = plt.Rectangle(
                        (MinimumX,-1.50*(Cart_Height/2 + Wheel_Radius*2)),
                        MaximumX-MinimumX,
                        0.50*(Cart_Height/2 + Wheel_Radius*2),
                        Color='0.70')
            ax0.add_patch(Ground)

            TimeStamp = ax0.text(
                0.75*MaximumX,
                0.75*1.50*(Cart_Height/2+Pendulum_Width/2+Pendulum_Length+Pendulum_Width/2),
                "Time: "+str(Time[0])+" s",
                color='0.50',
                fontsize=16
            )

            #Input

            Input, = ax1.plot([0],[self.U[0]],color = 'b')

            #Pendulum Angle

            Angle, = ax2.plot([0],[X2d[0]],color = 'r')

            #Cart Position

            Position, = ax3.plot([0],[self.X[0,0]],color = 'g')

            # Angular Velocity

            AngularVelocity, = ax4.plot([0],[X4d[0]],color = 'r')

            # Cart Velocity

            Velocity, = ax5.plot([0],[self.X[2,0]],color = 'g--')

            Markers.set_visible(False)
            SmallMarkers.set_visible(False)
            Cart.set_visible(False)
            FrontWheel.set_visible(False)
            FrontWheel_Rivet.set_visible(False)
            BackWheel.set_visible(False)
            BackWheel_Rivet.set_visible(False)
            Pendulum.set_visible(False)
            Pendulum_Attachment.set_visible(False)
            Pendulum_Rivet.set_visible(False)
            TimeStamp.set_visible(False)
            Ground.set_visible(True)
            Input.set_visible(False)
            Position.set_visible(False)
            Angle.set_visible(False)
            Velocity.set_visible(False)
            AngularVelocity.set_visible(False)

            return SmallMarkers,Markers,Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Ground,Input,Position,Angle,Velocity,TimeStamp,

        if self.dt <= 0.0001:
            framestep=2000
        elif self.dt <= 0.001:
            framestep=200
        elif self.dt <= 0.01:
            framestep=10
        else:
            framestep=1
        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(Time)-1,framestep),init_func=init, blit=False)
        if SaveAsGif==True:
            ani.save("visualizations_cart_pendulum/"+FileName+'.gif', writer='imagemagick', fps=10)
        plt.show()
    def return_all_params(self):
        return(
            {
                "States" : self.X,
                "Inputs" : self.U,
                "Costs" : self.TotalCost,
                "Angle Bounds" : [
                        min(self.X[0,:]),
                        max(self.X[0,:])
                    ],
                "X Position Bounds" : [
                        min(self.X[1,:]),
                        max(self.X[1,:])
                    ],
                "Angular Velocity Bounds" : [
                        min(self.X[2,:]),
                        max(self.X[2,:])
                    ],
                "X Velocity Bounds" : [
                        min(self.X[3,:]),
                        max(self.X[3,:])
                    ],
                "Input Bounds" : [
                        min(self.U),
                        max(self.U)
                    ],
                "V" : V,
                "Vx" : Vx,
                "Vxx" : Vxx,
                "Qu" : Qu,
                "Qx" : Qx,
                "Qux" : Qux,
                "Qxu" : Qxu,
                "Quu" : Quu,
                "Quu_inv" : Quu_inv,
                "Qxx" : Qxx
            }
        )

def cart_pole_DDP(X_o,**params):
    """
    Takes in the initial position (X_o) of the system along with an optional set of parameters (params), and runs DDP to meet some desired state.

    ####################################
    ############## params ##############
    ####################################

    1) Horizon - Number of timesteps into the future we wish to program. Must be an integer. Default is 300. (NOTE: Simulation ends at t = Horizon*dt)

    2) NumberOfIterations - Number of times to iterate the DDP. Must be an integer. Default is 100.

    3) dt - Discrete timestep. Must be either an int, float, float32, float64, or numpy.float. Default is 0.01. (NOTE: Simulation ends at t = Horizon*dt)

    4) U_o - Initial input to the system. Must be either None (meaning zero initial input to the system) or an array with shape (Horizon-1,). Default is None.

    5) p_target - Target state for the system to reach. Must be a (4,1) numpy matrix. Default is numpy.matrix([[0,0,0,0]]).T.

    6) LearningRate - rate at which the system converges to the new input. Must be either an int, float, float32, float64, or numpy.float and must be between 0 and 1. Default is 0.2.

    7) Q_f - Terminal cost matrix. Must be a (4,4) numpy matrix. Default is 50*numpy.matrix(numpy.eye(4)). Each element should be positive.

    8) R - Running cost scalar (only one input). Must be either an int, float, float32, float64, or numpy.float. Default is 0.001.

    9) PlotResults - Boolean to determine if the results of the program will be plotted. Default is False.

    10) AnimateResults - Boolean to determine if the results of the program will be animated. Default is False.

    11) ReturnAllResults - Boolean to determine if all results should be returned. Default is False. (NOTE: If False, the system will only return the values for the last iteration (X,U).)

    12) thresh - The cost threshold when the programmer will stop looking for a better solution . Must be a positive number (Default is 5).


    """
    #------------------------------------------------>
    #----------> Possible Parameter Values ---------->
    #------------------------------------------------>

    # Horizon - Number of timesteps into the future we wish to program
    Horizon = params.get("Horizon",300)
    assert type(Horizon)==int,\
        "Horizon must be an int, not "+str(type(Horizon))+". Default is 300."

    # NumberOfIterations - Number of times to iterate the DDP
    NumberOfIterations = params.get("NumberOfIterations",100)
    assert type(NumberOfIterations)==int, \
        "NumberOfIterations must be an int, not "+str(type(NumberOfIterations))+". Default is 100."

    # dt - Discrete timestep
    dt = params.get("dt",0.01)
    assert str(type(dt)) in [
            "<class 'int'>",
            "<class 'float'>",
            "<class 'float32'>",
            "<class 'float64'>",
            "<class 'numpy.float'>"], \
        "dt must be an int, float, float32, float64, or numpy.float not "+str(type(dt))+". Default is 0.01."

    # U_o - Initial input to the system.
    U_o = params.get("U_o",None)
    if U_o is None:
        U = np.zeros((Horizon-1,))
    else:
        assert np.shape(U_o)==(Horizon-1,), "U_o must be of shape ("+str(Horizon-1)+",) not "+str(np.shape(U_o))+"."
        U = U_o

    # p_target - Target state for the system to reach.
    p_target = params.get("p_target",np.matrix([[0,0,0,0]]).T)
    assert (str(type(p_target))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
            and np.shape(p_target)==(4,1)), \
        "p_target must be a (4,1) numpy matrix."

    # LearningRate - rate at which the system converges to the new input.
    LearningRate = params.get("LearningRate",0.2)
    assert (str(type(LearningRate)) in [
            "<class 'int'>",
            "<class 'float'>",
            "<class 'float32'>",
            "<class 'float64'>",
            "<class 'numpy.float'>"]
            and (0<LearningRate<=1)), \
        "LearningRate must be an int, float, float32, float64, or numpy.float not "+str(type(LearningRate))+", and should be between 0 and 1. Default is 0.2."

    # Q_f - Terminal cost matrix
    Q_f = params.get("Q_f",50*np.matrix(np.eye(4)))
    assert (str(type(Q_f))=="<class 'numpy.matrixlib.defmatrix.matrix'>"
            and np.shape(Q_f)==(4,4)), \
        "Q_f must be a (4,1) numpy matrix."

    # R - Running cost scalar (only one input).
    R = params.get("R",1e-3)
    assert (str(type(R)) in [
            "<class 'int'>",
            "<class 'float'>",
            "<class 'float32'>",
            "<class 'float64'>",
            "<class 'numpy.float'>"]
            and (R>0)), \
        "R must be an int, float, float32, float64, or numpy.float not "+str(type(R))+", and should be greater than 0. Default is 0.001."

    # PlotResults - Boolean to determine if the results of the program will be plotted.
    PlotResults = params.get("PlotResults",False)
    assert type(PlotResults)==bool, "PlotResults must be either True or False (Default)."

    # AnimateResults - Boolean to determine if the results of the program will be animated.
    AnimateResults = params.get("AnimateResults",False)
    assert type(AnimateResults)==bool, "AnimateResults must be either True or False (Default)."

    # ReturnAllResults - Boolean to determine if all of the results will be returned.
    ReturnAllResults = params.get("ReturnAllResults",False)
    assert type(ReturnAllResults)==bool, "ReturnAllResults must be either True or False (Default)."

    # thresh - Cost Threshold (must be a positive number)
    thresh = params.get("thresh",5)
    assert str(type(thresh)) in [
            "<class 'int'>",
            "<class 'float'>",
            "<class 'float32'>",
            "<class 'float64'>",
            "<class 'numpy.float'>"] and thresh > 0, \
        "thresh must be an int, float, float32, float64, or numpy.float not "+str(type(thresh))+" and must be positive. Default is 5."
    #------------------------------------------------>
    #-----------> Initializing the Problem ---------->
    #------------------------------------------------>

    # TotalX and TotalU
    TotalX = []
    TotalU = []

    # Correction array for input, not derivative of input.
    dU = np.zeros((Horizon-1,))

    # Initial trajectory:
    assert np.shape(X_o)==(4,), "X_o must have shape (4,) not "+str(np.shape(X_o))+"."
    X = forward_integrate_dynamics(X_o,U=U,Horizon=Horizon,dt=dt)

    TotalX.append(X)
    TotalU.append(U)

    V = [None]*Horizon # Each element must be a (4,1) matrix.
    Vx = [None]*Horizon # Each element must be a (4,1) matrix.
    Vxx = [None]*Horizon # Each element must be a (4,4) matrix.

    Qu = [None]*Horizon # Each element must be a (1,1) matrix.
    Qx = [None]*Horizon # Each element must be a (4,1) matrix.
    Qux = [None]*Horizon # Each element must be a (1,4) matrix.
    Qxu = [None]*Horizon # Each element must be a (4,1) matrix.
    Quu = [None]*Horizon # Each element must be a (1,1) matrix.
    Quu_inv = [None]*Horizon # Each element must be a (1,1) matrix.
    Qxx = [None]*Horizon # Each element must be a (4,4) matrix.

    TotalCost = [None]*NumberOfIterations

    # StartTime = time.time()
    statusbar = dsb(0,NumberOfIterations,title="Cart-Pole (DDP)")
    for k in range(NumberOfIterations):
        #------------------------------------------------>
        #--------> Linearization of the dynamics -------->
        #> Quadratic Approximations of the cost function >
        #------------------------------------------------>

        Phi,B = return_linearized_dynamics_matrices(X,U,dt)
        l,lx,lu,lux,lxu,luu,lxx = return_quadratic_cost_function_expansion_variables(X,U,R,dt)

        #------------------------------------------------>
        #--------------> Find the controls -------------->
        #------------------------------------------------>

        Vxx[Horizon-1]= Q_f
        Vx[Horizon-1] = Q_f * (np.matrix(X[:,Horizon-1]).T - p_target)
        V[Horizon-1] = (
            (np.matrix(X[:,Horizon-1]).T - p_target).T
            * Q_f
            * (np.matrix(X[:,Horizon-1]).T - p_target)
        )

        #------------------------------------------------>
        #----> Backpropagation of the Value Function ---->
        #------------------------------------------------>

        for j in reversed(range(Horizon-1)):
            Qx[j] = lx[j] + Phi[j].T * Vx[j+1]
            Qu[j] = lu[j] +  B[j].T * Vx[j+1]
            Qxu[j] = (
                lxu[j]
                + Phi[j].T * Vxx[j+1] * B[j]
            )
            Qux[j] = Qxu[j].T
            Quu[j] = (
                luu[j]
                + B[j].T * Vxx[j+1] * B[j]
            )
            Qxx[j] = (
                lxx[j]
                + Phi[j].T * Vxx[j+1] * Phi[j]
            )

            Quu_inv[j] = Quu[j]**(-1)

            Vxx[j] = Qxx[j] - Qxu[j] * Quu_inv[j] * Qux[j]
            Vx[j]= Qx[j] - Qxu[j] * Quu_inv[j] * Qu[j]
            # V[j] = l[j] + V[j+1] - 0.5 * Qu[j].T * Quu_inv[j] * Qu[j]
            V[j] = V[j+1] - 0.5*Qu[j].T * Quu_inv[j] * Qu[j]

        #------------------------------------------------>
        #-------------> Find the controlls -------------->
        #------------------------------------------------>

        U_new = np.zeros((Horizon-1,))
        dX = np.matrix(np.zeros((4,1)))
        for i in range(Horizon-1):
            dU = -Quu_inv[j]*(Qu[i] + Qux[i]*dX)
            dX = Phi[i]*dX + B[i]*dU
            U_new[i] = U[i] + LearningRate*dU

        U = U_new

        #------------------------------------------------>
        #-----> Simulation of the Nonlinear System ------>
        #------------------------------------------------>

        X = forward_integrate_dynamics(X_o,U=U,Horizon=Horizon,dt=dt)

        TotalX.append(X)
        TotalU.append(U)
        TotalCost[k] =  return_cost_for_a_given_trial(
                X,
                U,
                p_target,
                dt,
                Q_f,
                R
            )

        if TotalCost[k]<thresh:
            statusbar.update(NumberOfIterations)
        else:
            statusbar.update(k)
    #     print(
    #         'DDP Iteration %d,  Current Cost = %f \n'
    #         % (k+1,TotalCost[k])
    #     )
    #
    # print("Total Run Time: " + '%.2f'%(time.time()-StartTime) + "s")
    Endtime = Horizon*dt
    Time = np.arange(0,Endtime,dt)

    #------------------------------------------------------->
    #--------------------> Plot Results -------------------->
    #------------------------------------------------------->

    if PlotResults == True:
        plot_trajectory(Time,X,TotalCost,**params)

    #------------------------------------------------------->
    #------------------> Animate Results ------------------->
    #------------------------------------------------------->

    if AnimateResults == True:
        animate_trajectory(Time,X,U)

    #------------------------------------------------------->
    #-------------------> Return Results ------------------->
    #------------------------------------------------------->

    if ReturnAllResults == True:
        return(
            {
                "States" : TotalX,
                "Inputs" : TotalU,
                "Costs" : TotalCost,
                "params" : params,
                "Angle Bounds" : [
                        min(X[0,:]),
                        max(X[0,:])
                    ],
                "X Position Bounds" : [
                        min(X[1,:]),
                        max(X[1,:])
                    ],
                "Angular Velocity Bounds" : [
                        min(X[2,:]),
                        max(X[2,:])
                    ],
                "X Velocity Bounds" : [
                        min(X[3,:]),
                        max(X[3,:])
                    ],
                "Input Bounds" : [
                        min(U),
                        max(U)
                    ],
                "V" : V,
                "Vx" : Vx,
                "Vxx" : Vxx
            }
        )
    else:
        return(TotalX[-1],TotalU[-1])
