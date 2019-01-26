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
from params import *
from quadratic_cost_function_expansion_variables import *
from linearized_dynamics import *
from feedforward_simulation import *
from cost_function import *
from animate import *
import time

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

    StartTime = time.time()
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
            Qux[j] = lux[j] + B[j].T * Vxx[j+1] * Phi[j]
            Qxu[j] = lxu[j] + Phi[j].T * Vxx[j+1] * B[j]
            Quu[j] = luu[j] + B[j].T * Vxx[j+1] * B[j]
            Qxx[j] = lxx[j] + Phi[j].T * Vxx[j+1] * Phi[j]

            Quu_inv[j] = Quu[j]**(-1)

            Vxx[j] = Qxx[j] - Qxu[j] * Quu_inv[j] * Qux[j]
            Vx[j]= Qx[j] - Qxu[j] * Quu_inv[j] * Qu[j]
            # V[j] = l[j] + V[j+1] - 0.5 * Qu[j].T * Quu_inv[j] * Qu[j]
            V[j] = V[j+1] - Qu[j].T * Quu_inv[j] * Qu[j]

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

        print(
            'DDP Iteration %d,  Current Cost = %f \n'
            % (k+1,TotalCost[k])
        )

    print("Total Run Time: " + '%.2f'%(time.time()-StartTime) + "s")
    Endtime = Horizon*dt
    Time = np.arange(0,Endtime,dt)

    #------------------------------------------------------->
    #--------------------> Plot Results -------------------->
    #------------------------------------------------------->

    if PlotResults == True:
        fig1 = plt.figure(figsize=(12,8))
        plt.suptitle("Cart Pole Control via DDP",fontsize=16)

        plt.subplot(321)
        plt.plot(
            Time,
            p_target[0,0]*np.ones((Horizon,)),
            'r--',
            linewidth=2
        )
        plt.plot(Time,X[0,:],linewidth=4)
        # plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('X Position',fontsize=16)
        plt.grid(True)

        plt.subplot(322)
        plt.plot(
            Time,
            p_target[1,0]*np.ones((Horizon,)),
            'r--',
            linewidth=2
        )
        plt.plot(Time,X[1,:],linewidth=4)
        # plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('Theta',fontsize=16)
        plt.grid(True)

        plt.subplot(323)
        plt.plot(
            Time,
            p_target[2,0]*np.ones((Horizon,)),
            'r--',
            linewidth=4
        )
        plt.plot(Time,X[2,:],linewidth=4)
        # plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('X velocity',fontsize=16)
        plt.grid(True)

        plt.subplot(324)
        plt.plot(
            Time,
            p_target[3,0]*np.ones((Horizon,)),
            'r--',
            linewidth=4
        )
        plt.plot(Time,X[3,:],linewidth=4)
        # plt.xlabel('Time (s)',fontsize=16)
        plt.ylabel('Angular Velocity',fontsize=16)
        plt.grid(True)

        plt.subplot(3,2,(5,6))
        plt.plot(TotalCost,linewidth=2)
        plt.xlabel('Iterations',fontsize=16)
        plt.ylabel('Cost',fontsize=16)

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
                    ]
            }
        )
    else:
        return(TotalX[-1],TotalU[-1])
