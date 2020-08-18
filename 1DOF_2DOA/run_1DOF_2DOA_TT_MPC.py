#######################################################
#######################################################
#############  Model Predictive Control  ##############
####   for Cart - Dbl Pole Rigid Body Dynamics    #####
#######################################################
#######################################################
##                                                   ##
## Course: DDP Derivation and Control System Studies ##
##              Author: Hagen, Daniel                ##
##                                                   ##
#######################################################
#######################################################

import numpy as np
from params import *
from class_1DOF_2DOA_TT_DDP import *
from danpy.sb import dsb
from animate import *
from plot import *
from useful_functions import *
import random

NumberOfIterations = 1
params["dt"] = 0.01
params["Horizon"] = int(0.1/params["dt"])
params["Time Duration"] = 5

#########################################
#########################################
############# Testing Class #############
#########################################
#########################################

X = np.zeros((2,int(params["Time Duration"]/params["dt"]) + 1))
U = np.zeros((2,int(params["Time Duration"]/params["dt"])))
TotalCost = np.zeros((int(params["Time Duration"]/params["dt"]),))

params["NumberOfIterations"] = NumberOfIterations
# params["p_target"] = np.matrix([[5,0,0,0,0,0]]).T

AbsoluteHorizon = params["Horizon"]

params["LearningRate"] = 0.2 # change to 1 if using Tassa et al equation

X_o = np.array([np.pi/12,0])
X[:,0] = X_o

Time = np.arange(0,params["Time Duration"]+params["dt"],params["dt"])

thresh1 = 1

statusbar = dsb(0,int(params["Time Duration"]/params["dt"]),
        title="1 DOF, 2 DOA Pendulum (MPC)"
    )
## --> Initialize DDP from initial state (X_o) with Horizon = Horizon and U_o = np.zeros((Horizon,))
DDP = DDP_1DOF_2DOA(X_o,**params)
DDP.find_initial_input(Seed=None) # Only run if X_o is not [0,_]
U[:,0] = DDP.U[:,0]
for i in range(int(params["Time Duration"]/params["dt"])):
    ## --> Run DDP from (current) initial state (X_o) with Horizon = (Horizon-i)
    DDP.run_ddp()

    # ## --> Update Horizon to be 1 step closer than previous
    # DDP.set_Horizon((AbsoluteHorizon-i)-1)

    ## --> Update initial state (X_o) to second state of previous DDP and append states list.
    DDP.set_X_o(DDP.X[:,1])
    X[:,i+1] = DDP.X[:,1]
    # if Time[i]>5 and random.random()>0.2*params["dt"]:
    #     DDP.set_X_o(DDP.X[:,1]+[0,0,10*np.pi/180,0])
    #     X[:,i+1] = DDP.X[:,1]+[0,0,10*np.pi/180,0]

    ## --> Change initial input (U_o) to be final input of the previous DDPs iteration.
    if i < params["Time Duration"]/params["dt"] - 1:
        DDP.set_U_o(np.concatenate([DDP.U[:,1:],DDP.U[:,np.newaxis,-1]],axis=1))
        # DDP.set_U_o(np.concatenate([DDP.U[1:],[0]]))
        # DDP.set_U_o(np.zeros(np.shape(DDP.U)))
        # DDP.set_U_o(DDP.U[1]*np.ones(np.shape(DDP.U)))
        U[:,i+1] = DDP.U[:,0]

    ## --> Update cost array to be the last cost value of the previous DDP
    TotalCost[i] = DDP.TotalCost[-1]

    # ## --> Break if cost is below threshold.
    # if TotalCost[i]<thresh1:
    #     X[:,(i+2):] = DDP.X[:,2:]
    #     U[(i+2):] = DDP.U[1:]
    #     # [X.append(DDP.X[:,j,np.newaxis]) for j in range(2,np.shape(DDP.X)[1])]
    #     # [U.append(u) for u in DDP.U[1:]]
    #     statusbar.update((AbsoluteHorizon-1)-1)
    #     break
    # else:
    #     statusbar.update(i)
    statusbar.update(i)

# params["Horizon"] = AbsoluteHorizon

#########################################
#########################################
########### Plotting Results ############
#########################################
#########################################

fig1 = plot_trajectory(Time,X,U,TotalCost,ReturnFig=True,**params)
FilePath = save_figures("visualizations/1DOF_2DOA_TT/MPC/","v1.0",params,ReturnPath=True,SaveAsPDF=True)
animate_trajectory(Time,X,U,SaveAsGif=True,FileName=FilePath+"1DOF_2DOA_TT_MPC")

del(params)
