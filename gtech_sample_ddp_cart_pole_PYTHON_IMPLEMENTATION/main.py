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
from cost_function import *
from State_And_Control_Transition_Matrices import *
from fnsimulate import *
from fnCostComputation import *
from animate import *

# global m  #Changed the initial paramaters to reflect the new problem.
# global M
# global l
# global gr
# global h

# h is the step used to determine the derivative
h = 0.000001

# Horizon
Horizon = 300 # 1.5sec
# Number of Iterations
num_iter = 100

# Discretization
dt = 0.01
global timeee
timeee = Horizon*dt

# Weight in Final State:
Q_f = np.matrix(np.zeros((4,4)))
Q_f[0,0] = 5
Q_f[1,1] = 1000
Q_f[2,2] = 5
Q_f[3,3] = 50

# Weight in the Control:
# Modified from original because our control is only one dimensional.
R = 1e-3

# Initial Configuration:
xo = np.zeros((4,))
xo[0] = 0
xo[1] = np.pi

# Initial Control:
u_k = np.zeros((Horizon-1,))
du_k = np.zeros((Horizon-1,))


# Initial trajectory:
x_traj = np.zeros((4,Horizon))


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
gamma = 0.2

q0 = np.zeros((1,Horizon))
q_k = np.zeros((4,1,Horizon))
Q_k = np.zeros((4,4,Horizon))
r_k = np.zeros((1,Horizon))
R_k = np.zeros((1,1,Horizon))
P_k = np.zeros((1,4,Horizon))

A = np.zeros((4,4,Horizon))
B = np.zeros((4,1,Horizon))

V = np.zeros((1,Horizon))
Vx = np.zeros((4,1,Horizon))
Vxx = np.zeros((4,4,Horizon))

l_k = np.zeros((1,Horizon))
L_k = np.zeros((1,4,Horizon))

Cost = np.zeros((num_iter,))

for k in range(num_iter):
    #------------------------------------------------>
    #--------> Linearization of the dynamics -------->
    #> Quadratic Approximations of the cost function >
    #------------------------------------------------>

    for  j in range(Horizon-1):

        l0,l_x,l_xx,l_u,l_uu,l_ux = fnCost(
                x_traj[:,j],
                u_k[j],
                j,
                R,
                dt
            );
        q0[:,j] = dt * l0
        q_k[:,:,j] = dt * l_x
        Q_k[:,:,j] = dt * l_xx
        r_k[:,j] = dt * l_u
        R_k[:,:,j] = dt * l_uu
        P_k[:,:,j] = dt * l_ux

        dfx,dfu = fnState_And_Control_Transition_Matrices(
                x_traj[:,j],
                u_k[j],
                du_k[j],
                dt
            )

        A[:,:,j] = np.eye(4) + dfx * dt
        B[:,:,j] = dfu * dt

    #------------------------------------------------>
    #-------------> Find the controlls -------------->
    #------------------------------------------------>

    Vxx[:,:,Horizon-1]= Q_f
    Vx[:,:,Horizon-1] = Q_f * (np.matrix(x_traj[:,Horizon-1]).T - p_target)
    V[:,Horizon-1] = (
        (np.matrix(x_traj[:,Horizon-1]).T - p_target).T
        * Q_f
        * (np.matrix(x_traj[:,Horizon-1]).T - p_target)
    )

    #------------------------------------------------>
    #----> Backpropagation of the Value Function ---->
    #------------------------------------------------>

    for j in reversed(range(Horizon-1)):

        H = (
            np.matrix(R_k[:,:,j])
            + np.matrix(B[:,:,j]).T * np.matrix(Vxx[:,:,j+1]) * np.matrix(B[:,:,j])
        )
        G = (
            np.matrix(P_k[:,:,j])
            + np.matrix(B[:,:,j]).T * np.matrix(Vxx[:,:,j+1]) * np.matrix(A[:,:,j])
        )
        g = np.matrix(r_k[:,j]) +  np.matrix(B[:,:,j]).T * np.matrix(Vx[:,:,j+1])


        inv_H = H**(-1)
        L_k[:,:,j] = - inv_H[0,0] * G
        l_k[:,j] = - inv_H[0,0] * g


        Vxx[:,:,j] = (
            np.matrix(Q_k[:,:,j])
            + np.matrix(A[:,:,j]).T * np.matrix(Vxx[:,:,j+1]) * np.matrix(A[:,:,j])
            + np.matrix(L_k[:,:,j]).T * H * np.matrix(L_k[:,:,j])
            + np.matrix(L_k[:,:,j]).T * G
            + G.T * np.matrix(L_k[:,:,j])
        )
        Vx[:,:,j]= (
            np.matrix(q_k[:,:,j])
            + np.matrix(A[:,:,j]).T * np.matrix(Vx[:,:,j+1])
            + np.matrix(L_k[:,:,j]).T * g
            + G.T * l_k[:,j]
            + np.matrix(L_k[:,:,j]).T * H * l_k[:,j]
        )
        V[:,j] = (
            q0[:,j]
            + V[:,j+1]
            + 0.5 * np.matrix(l_k[:,j]).T * H * np.matrix(l_k[:,j])
            + np.matrix(l_k[:,j]).T * g
        )

    #------------------------------------------------>
    #-------------> Find the controlls -------------->
    #------------------------------------------------>

    u_new = np.zeros((Horizon,))
    dx = np.matrix(np.zeros((4,1)))
    for i in range(Horizon-1):
        du = np.matrix(l_k[:,i]) + np.matrix(L_k[:,:,i])*dx
        dx = np.matrix(A[:,:,i])*dx + np.matrix(B[:,:,i])*du
        u_new[i] = u_k[i] + gamma*du

    u_k = u_new

    #------------------------------------------------>
    #-----> Simulation of the Nonlinear System ------>
    #------------------------------------------------>

    x_traj = fnsimulate(xo,u_new,Horizon,dt)
    Cost[k] =  fnCostComputation(
            x_traj,
            u_k,
            p_target,
            dt,
            Q_f,
            R
        )
    # x1[k,:] = x_traj[0,:]


    print(
        'iLQG Iteration %d,  Current Cost = %f \n'
        % (k+1,Cost[k])
    )

Endtime = Horizon*dt
Time = np.arange(0,Endtime,dt)

#------------------------------------------------------->
#--------------------> Plot Section -------------------->
#------------------------------------------------------->

fig1 = plt.figure(figsize=(12,8))

plt.subplot(321)
plt.plot(
    Time,
    p_target[0,0]*np.ones((Horizon,)),
    'r--',
    linewidth=2
)
plt.plot(Time,x_traj[0,:],linewidth=4)
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
plt.plot(Time,x_traj[1,:],linewidth=4)
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
plt.plot(Time,x_traj[2,:],linewidth=4)
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
plt.plot(Time,x_traj[3,:],linewidth=4)
# plt.xlabel('Time (s)',fontsize=16)
plt.ylabel('Angular Velocity',fontsize=16)
plt.grid(True)

plt.subplot(3,2,(5,6))
plt.plot(Cost,linewidth=2)
plt.xlabel('Iterations',fontsize=16)
plt.ylabel('Cost',fontsize=16)

animate_trajectory(Time,x_traj,u_k)
# run('animate.m')
Output ={
    "Angle Bounds" : [
            min(x_traj[0,:]),
            max(x_traj[0,:])
        ],
    "X Position Bounds" : [
            min(x_traj[1,:]),
            max(x_traj[1,:])
        ],
    "Angular Velocity Bounds" : [
            min(x_traj[2,:]),
            max(x_traj[2,:])
        ],
    "X Velocity Bounds" : [
            min(x_traj[3,:]),
            max(x_traj[3,:])
        ],
    "Input Bounds" : [
            min(u_k),
            max(u_k)
        ]
}
