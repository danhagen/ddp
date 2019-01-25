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

q0 = [None]*Horizon # Scalar list
q_k = [None]*Horizon # Each element must be a (4,1) matrix. Place assertion function or create test.
Q_k = [None]*Horizon # Each element must be a (4,4) matrix. Place assertion function or create test.
r_k = [None]*Horizon # Scalar list
R_k = [None]*Horizon # Scalar list
P_k = [None]*Horizon # Each element must be a (1,4) matrix. Place assertion function or create test.

Phi = [None]*Horizon # Each element must be a (4,4) matrix. Place assertion function or create test.
B = [None]*Horizon # Each element must be a (4,4) matrix. Place assertion function or create test.

V = [None]*Horizon # Scalar list
Vx = [None]*Horizon # Each element must be a (4,1) matrix. Place assertion function or create test.
Vxx = [None]*Horizon # Each element must be a (4,4) matrix. Place assertion function or create test.

l_k = [None]*Horizon # Scalar list
L_k = [None]*Horizon # Each element must be a (1,4) matrix. Place assertion function or create test.

TotalCost = [None]*NumberOfIterations

StartTime = time.time()
for k in range(NumberOfIterations):
    #------------------------------------------------>
    #--------> Linearization of the dynamics -------->
    #> Quadratic Approximations of the cost function >
    #------------------------------------------------>

    Phi,B = return_linearized_dynamics_matrices(X,U,dt)
    l,lx,lu,lux,lxu,luu,lxx = return_quadratic_cost_function_expansion_variables(X,U,dt)
    q0 = l
    q_k = lx
    Q_k = lxx
    r_k = lu
    R_k = luu
    P_k = lux

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

        H = R_k[j] + B[j].T*Vxx[j+1]*B[j]
        G = P_k[j] + B[j].T*Vxx[j+1]*Phi[j]
        g = r_k[j] +  B[j].T * Vx[j+1]


        inv_H = H**(-1)
        L_k[j] = - inv_H[0,0] * G
        l_k[j] = - inv_H[0,0] * g


        Vxx[j] = (
            Q_k[j]
            + Phi[j].T * Vxx[j+1] * Phi[j]
            + L_k[j].T * H * L_k[j]
            + L_k[j].T * G
            + G.T * L_k[j]
        )
        Vx[j]= (
            q_k[j]
            + Phi[j].T * Vx[j+1]
            + L_k[j].T * g
            + G.T * l_k[j]
            + L_k[j].T * H * l_k[j]
        )
        V[j] = (
            q0[j]
            + V[j+1]
            + 0.5 * l_k[j].T * H * l_k[j]
            + l_k[j].T * g
        )

    #------------------------------------------------>
    #-------------> Find the controlls -------------->
    #------------------------------------------------>

    U_new = np.zeros((Horizon-1,))
    dX = np.matrix(np.zeros((4,1)))
    for i in range(Horizon-1):
        dU = l_k[i] + L_k[i]*dX
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
plt.suptitle("Cart Pole Control via iLQG",fontsize=16)

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
