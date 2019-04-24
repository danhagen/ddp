import matplotlib.pyplot as plt
import numpy as np
from params import *

def plot_trajectory(Time,X,TotalCost,**params):
    p_target = params["p_target"]
    Horizon = params["Horizon"]

    fig1 = plt.figure(figsize=(12,8))
    plt.suptitle("Cart Pole Control via DDP",fontsize=16)

    plt.subplot(321)
    plt.plot(
        [Time[0],Time[-1]],
        [p_target[0,0]]*2,
        'r--',
        linewidth=2
    )
    plt.plot(Time,X[0,:],linewidth=4)
    # plt.xlabel('Time (s)',fontsize=16)
    plt.ylabel('X Position',fontsize=16)
    plt.grid(True)

    plt.subplot(322)
    plt.plot(
        [Time[0],Time[-1]],
        [p_target[1,0]]*2,
        'r--',
        linewidth=2
    )
    plt.plot(Time,X[1,:],linewidth=4)
    # plt.xlabel('Time (s)',fontsize=16)
    plt.ylabel('Theta',fontsize=16)
    plt.grid(True)

    plt.subplot(323)
    plt.plot(
        [Time[0],Time[-1]],
        [p_target[2,0]]*2,
        'r--',
        linewidth=4
    )
    plt.plot(Time,X[2,:],linewidth=4)
    # plt.xlabel('Time (s)',fontsize=16)
    plt.ylabel('X velocity',fontsize=16)
    plt.grid(True)

    plt.subplot(324)
    plt.plot(
        [Time[0],Time[-1]],
        [p_target[3,0]]*2,
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
