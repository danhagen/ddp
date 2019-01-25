#######################################################
#######################################################
#######  Iterative Linear Quadratic Regulator  ########
#######   for Cart-Pole Rigid Body Dynamics    ########
#######################################################
#######################################################
##                                                   ##
## Course: DDP Derivation and Control System Studies ##
## Author: Evangelos, Theodorou                      ##
## Adapted for PYTHON by: Hagen, Daniel              ##
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

# Horizon
Horizon = 300 # 1.5sec

# Number of Iterations
NumberOfIterations = 100

# Discretization
dt = 0.01

# TotalX and TotalU
TotalX = []
TotalU = []

# Initial Configuration:
X_o = np.zeros((4,))
X_o[0] = 0
X_o[1] = np.pi

# Initial Control:
U = np.zeros((Horizon-1,))
dU = np.zeros((Horizon-1,))

# Initial trajectory:
X = forward_integrate_dynamics(
        X_o,
        U=U,
        Horizon=Horizon,
        dt=dt)

TotalX.append(X)
TotalU.append(U)

# Target:
p_target = np.matrix(
    [
        [0],
        [0],
        [0],
        [0]
    ]
)

# Learning Rate:
LearningRate = 0.2

V = [None]*Horizon # Scalar list
Vx = [None]*Horizon # Each element must be a (4,1) matrix. Place assertion function or create test.
Vxx = [None]*Horizon # Each element must be a (4,4) matrix. Place assertion function or create test.

def return_empty_lists_for_quadratic_expansion_of_Q(length):
    Qu = [None]*length
    Qx = [None]*length
    Qux = [None]*length
    Qxu = [None]*length
    Quu = [None]*length
    Quu_inv = [None]*length
    Qxx = [None]*length
    return(Qu,Qx,Qux,Qxu,Quu,Quu_inv,Qxx)

TotalCost = [None]*NumberOfIterations

StartTime = time.time()
for k in range(NumberOfIterations):
    #------------------------------------------------>
    #--------> Linearization of the dynamics -------->
    #> Quadratic Approximations of the cost function >
    #------------------------------------------------>

    Phi,B = return_linearized_dynamics_matrices(X,U,dt)
    l,lx,lu,lux,lxu,luu,lxx = return_quadratic_cost_function_expansion_variables(X,U,dt)

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

    Qu,Qx,Qux,Qxu,Quu,Quu_inv,Qxx = return_empty_lists_for_quadratic_expansion_of_Q(Horizon)
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
        V[j] = l[j] + V[j+1] - 0.5 * Qu[j].T * Quu_inv[j] * Qu[j]

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
    X = forward_integrate_dynamics(
            X_o,
            U=U,
            Horizon=Horizon,
            dt=dt)

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
        'iLQG Iteration %d,  Current TotalCost = %f \n'
        % (k+1,TotalCost[k])
    )

print("Total Run Time: " + '%.2f'%(time.time()-StartTime) + "s")
Endtime = Horizon*dt
Time = np.arange(0,Endtime,dt)

#------------------------------------------------------->
#--------------------> Plot Section -------------------->
#------------------------------------------------------->

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
plt.ylabel('TotalCost',fontsize=16)

animate_trajectory(Time,X,U)
# run('animate.m')
Output ={
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
