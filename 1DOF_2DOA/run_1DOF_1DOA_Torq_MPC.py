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
from run_1DOF_1DOA_Torq_DDP import *
from danpy.sb import dsb
from animate import *
from plot import *
from useful_functions import *
import random

NumberOfIterations = 1
params["dt"] = 0.01
params["Horizon"] = int(0.2/params["dt"])
params["Time Duration"] = 5

#########################################
#########################################
############# Testing Class #############
#########################################
#########################################

X = np.zeros((2,int(params["Time Duration"]/params["dt"]) + 1))
U = np.zeros(int(params["Time Duration"]/params["dt"]),)
TotalCost = np.zeros((int(params["Time Duration"]/params["dt"]),))

params["NumberOfIterations"] = NumberOfIterations
params["p_target"] = np.matrix([[np.pi/4,0]]).T

params["U_o"] = np.zeros((params["Horizon"]-1,))
params["LearningRate"] = 0.2

X_o = np.array([0,0])
X[:,0] = X_o

Time = np.arange(0,params["Time Duration"]+params["dt"],params["dt"])

statusbar = dsb(0,int(params["Time Duration"]/params["dt"]),
        title="1 DOF, 1 DOA Pendulum (MPC)"
    )
## --> Initialize DDP from initial state (X_o) with Horizon = Horizon and U_o = np.zeros((Horizon,))
DDP = DDP_1DOF_1DOA_Torq(X_o,**params)
TotalTime = 0
test = []
for i in range(int(params["Time Duration"]/params["dt"])):
    ## --> Run DDP from (current) initial state (X_o) with Horizon = (Horizon-i)
    StartTime = time.time()
    DDP.run_ddp()
    TotalTime += (time.time()-StartTime)
    ## --> Update initial state (X_o) to second state of previous DDP and append states list.
    DDP.set_X_o(DDP.X[:,1])
    X[:,i+1] = DDP.X[:,1]
    # if Time[i]>5 and random.random()>0.2*params["dt"]:
    #     DDP.set_X_o(DDP.X[:,1]+[0,0,10*np.pi/180,0])
    #     X[:,i+1] = DDP.X[:,1]+[0,0,10*np.pi/180,0]

    ## --> Change initial input (U_o) to be final input of the previous DDPs iteration.
    if i < params["Time Duration"]/params["dt"] - 1:
        DDP.set_U_o(np.concatenate([DDP.U[np.newaxis,1:].T,np.array([DDP.U[-1]],ndmin=2)]).squeeze())
        # test.append(np.concatenate([DDP.U[np.newaxis,:-1].T,np.array([DDP.U[-1]],ndmin=2)]).squeeze())
        # DDP.set_U_o(np.concatenate([DDP.U[1:],[0]]))
        # DDP.set_U_o(np.zeros(np.shape(DDP.U)))
        # DDP.set_U_o(DDP.U[1]*np.ones(np.shape(DDP.U)))
        U[i+1] = DDP.U[0]

    ## --> Update cost array to be the last cost value of the previous DDP
    TotalCost[i] = DDP.TotalCost[-1]

    statusbar.update(i)

print("Time per DDP (" + str(params["NumberOfIterations"]) + " Iterations each): " + str(TotalTime/(int(params["Time Duration"]/params["dt"]))) + " sec.")

#########################################
#########################################
########### Plotting Results ############
#########################################
#########################################

U_new = np.concatenate([U[np.newaxis,:],np.zeros((1,len(U)))])
fig1 = plot_trajectory(Time,X,U_new,TotalCost,ReturnFig=True,**params)
FilePath = save_figures("visualizations/1DOF_1DOA_Torq/MPC/","v1.0",params,ReturnPath=True,SaveAsPDF=True)
animate_trajectory(Time,X,U_new,SaveAsGif=True,FileName=FilePath+"1DOF_1DOA_Torq_MPC")

# del(params)
