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

X_o = [np.pi/12,0]
DDP = DDP_1DOF_2DOA(X_o,**params)
DDP.find_initial_input(Seed=None) # RUN ONLY IF X_o != [0,0] as R1([0,_])=0
DDP.run_ddp()
fig1 = DDP.plot_trajectory(ReturnFig=True)
FilePath = save_figures("visualizations/1DOF_2DOA_TT/DDP/","v1.0",params,ReturnPath=True,SaveAsPDF=True)
DDP.animate_trajectory(SaveAsGif=True,FileName=FilePath+"1DOF_2DOA_TT_DDP")
