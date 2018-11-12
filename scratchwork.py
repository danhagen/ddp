import numpy as np
import matplotlib.pyplot as plt
from danpy.sb import *
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy import signal

"""
Notes:

X1: angle of pendulum
X2: angular velocity of pendulum

"""
m = 1 # kg
L = 0.5 # m
g = 9.8 # m/s²
b = 0 # Nms

TargetAngle = np.pi #in radians
TargetAngularVelocity = 0 #in radians

t1 = 1000
t2 = 1000
k1 = 1000
k2 = 1000

def dx1_dt(X,U):
    return(X[1])
def dx2_dt(X,U):
    return(-(g/L)*np.sin(X[0]) - b/(m*L**2)*X[1] + (1/(m*L**2))*U)

def forward_integrate_dynamics(
        ICs,
        UsingDegrees=True,
        Animate=False,
        U=None,
        ReturnX=False,
        **kwargs):
    """
    ICs must be a list of floats and/or ints of length 2.

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **kwargs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    UsingDegrees must be a bool. Default is True. If True, then the ICs for pendulum angle and angular velocity can be given in degrees and degrees per second, respectively.

    Animate must be a bool. Default is False. If True, the program will run animate_trajectory().

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Notes:

    X1: angle of pendulum
    X2: angular velocity of pendulum

    """
    assert type(ICs)==list and len(ICs)==2, "ICs must be a list of length 2."
    LocationStrings = ["1st", "2nd"]
    for i in range(2):
        assert str(type(ICs[i])) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>"],\
            "ICs must be numbers. Check the " + LocationString[i] + " element of IC"
    assert type(UsingDegrees)==bool, "UsingDegrees must be either True or False."
    assert type(Animate)==bool, "Animate must be either True or False."

    dt = kwargs.get("dt",0.0001)
    assert str(type(dt)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>"],\
        "dt must be a number."

    N_seconds = kwargs.get("N_seconds",10)
    assert str(type(N_seconds)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>"],\
        "N_seconds must be a number."

    Time = np.arange(0,N_seconds+dt,dt)
    X = np.zeros((2,len(Time)))
    if U is None:
        U = np.zeros(len(Time))
    else:
        assert len(U)==len(Time)-1, "U must have length = (len(Time)-1)."

    # ICs

    if UsingDegrees:
        X[0,0] = ICs[0]*(np.pi/180)
        X[1,0] = ICs[1]*(np.pi/180)
    else:
        X[0,0] = ICs[0]
        X[1,0] = ICs[1]


    for i in range(len(Time)-1):
        X[0,i+1] = X[0,i] + dx1_dt(X[:,i],U[i])*dt
        X[1,i+1] = X[1,i] + dx2_dt(X[:,i],U[i])*dt


    if ReturnX==True:
        return(X)
    else:
        if Animate:
            animate_trajectory(Time,X,U)
        else:
            plt.figure(figsize=(15,4))

            ax1 = plt.subplot(121)
            ax1.plot(Time,180*X[0,:]/np.pi,'g')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Pendulum Angle (deg)')
            if max(abs(180*X[0,:]/np.pi - 180*X[0,0]/np.pi))<1e-7:
                ax1.set_ylim([180*X[0,0]/np.pi - 5,180*X[0,0]/np.pi + 5])

            ax2 = plt.subplot(122)
            ax2.plot(Time,180*X[1,:]/np.pi,'g--')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Pendulum Angular \n Velocity (deg/s)')
            if max(abs(X[1,:]-X[1,0]))<1e-7:
                ax2.set_ylim([X[1,0]-1,X[1,0]+1])

            plt.show()

def animate_trajectory(Time,X,U,SaveAsGif=False):

        # Angles must be in degrees for animation

    X1d = X[0,:]*(180/np.pi)
    X2d = X[1,:]*(180/np.pi)


    fig = plt.figure(figsize=(15,12))
    ax1 = plt.subplot2grid((2,6),(0,0),colspan=6) # animation
    ax2 = plt.subplot2grid((2,6),(1,0),colspan=2) # pendulum angle
    ax3 = plt.subplot2grid((2,6),(1,2),colspan=2) # pendulum angular velocity
    ax4 = plt.subplot2grid((2,6),(1,4),colspan=2)

    plt.suptitle("Simple Pendulum Example",Fontsize=28,y=0.95)

    Pendulum_Width = 0.5*L
    Pendulum_Length = 2*L

    # Model Drawing

    Pendulum, = ax1.plot(
                    [
                        0,
                        Pendulum_Length*np.sin(X[0,0])
                    ],
                    [
                        0,
                         -Pendulum_Length*np.cos(X[0,0])
                    ],
                    Color='0.50',
                    lw = 10,
                    solid_capstyle='round'
                    )


    Pendulum_Attachment = plt.Circle((0,0),2*Pendulum_Width/2,Color='#4682b4')
    ax1.add_patch(Pendulum_Attachment)

    Pendulum_Rivet, = ax1.plot(
        [0],
        [0],
        c='k',
        marker='o',
        lw=2
        )

    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax1.set_frame_on(True)

    ax1.set_xlim(
        [
            -1.20*Pendulum_Length,
            1.20*Pendulum_Length
        ]
    )
    ax1.set_ylim(
        [
            -1.20*Pendulum_Length,
            1.20*Pendulum_Length
        ]
        )
    ax1.set_aspect('equal')

    #Pendulum Angle

    Angle, = ax2.plot([0],[X1d[0]],color = 'r')
    ax2.set_xlim(0,Time[-1])
    ax2.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax2.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X1d-X1d[0]))<1e-7:
        ax2.set_ylim([X1d[0]-2,X1d[0]+2])
    else:
        RangeX1d= max(X1d)-min(X1d)
        ax2.set_ylim([min(X1d)-0.1*RangeX1d,max(X1d)+0.1*RangeX1d])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title("Pendulum Angle (deg)",fontsize=16,fontweight = 4,color = 'r',y = 1.05)

    # Angular Velocity

    AngularVelocity, = ax3.plot([0],[X2d[0]],color='r',linestyle='--')
    ax3.set_xlim(0,Time[-1])
    ax3.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax3.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X2d-X2d[0]))<1e-7:
        ax3.set_ylim([X2d[0]-2,X2d[0]+2])
    else:
        RangeX2d= max(X2d)-min(X2d)
        ax3.set_ylim([min(X2d)-0.1*RangeX2d,max(X2d)+0.1*RangeX2d])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_title("Pendulum Angular Velocity (deg/s)",fontsize=16,fontweight = 4,color = 'r',y = 1.05)

    # Input Torque

    Torque, = ax4.plot([0],[U[0]],color='g')
    ax4.set_xlim(0,Time[-1])
    ax4.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax4.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(U-U[0]))<1e-7:
        ax4.set_ylim([U[0]-2,U[0]+2])
    else:
        RangeU= max(U)-min(U)
        ax4.set_ylim([min(U)-0.1*RangeU,max(U)+0.1*RangeU])
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_title("Applied Torque (Nm)",fontsize=16,fontweight = 4,color = 'g',y = 1.05)

    def animate(i):

        Pendulum.set_xdata([0, Pendulum_Length*np.sin(X[0,i])])
        Pendulum.set_ydata([0,-Pendulum_Length*np.cos(X[0,i])])

        Angle.set_xdata(Time[:i])
        Angle.set_ydata(X1d[:i])

        AngularVelocity.set_xdata(Time[:i])
        AngularVelocity.set_ydata(X2d[:i])

        Torque.set_xdata(Time[:i])
        Torque.set_ydata(U[:i])

        return Pendulum,Angle,AngularVelocity,Torque,

    # Init only required for blitting to give a clean slate.
    def init():

        Pendulum, = ax1.plot(
                        [
                            0,
                            Pendulum_Length*np.sin(X[0,0])
                        ],
                        [
                            0,
                            -Pendulum_Length*np.cos(X[1,0])
                        ],
                        Color='0.50',
                        lw = 10,
                        solid_capstyle='round'
                        )

        Pendulum_Attachment = plt.Circle((0,0),2*Pendulum_Width/2,Color='#4682b4')
        ax1.add_patch(Pendulum_Attachment)

        Pendulum_Rivet, = ax1.plot(
            [0],
            [0],
            c='k',
            marker='o',
            lw=2
            )

        #Pendulum Angle

        Angle, = ax2.plot([0],[X1d[0]],color = 'r')

        # Angular Velocity

        AngularVelocity, = ax3.plot([0],[X2d[0]],color = 'r')

        # Input Torque

        Torque, = ax4.plot([0],[U[0]],color='g')

        Pendulum.set_visible(False)
        Pendulum_Attachment.set_visible(False)
        Pendulum_Rivet.set_visible(False)
        Angle.set_visible(False)
        AngularVelocity.set_visible(False)
        Torque.set_visible(False)

        return Pendulum,Pendulum_Attachment,Pendulum_Rivet,Angle,AngularVelocity,Torque

    dt = Time[1]-Time[0]
    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(Time)-1,10), init_func=init, blit=False)
    if SaveAsGif==True:
        ani.save('simple_pendulum_ddp.gif', writer='imagemagick', fps=10)
    plt.show()

def return_Phi(X,U,dt):
    return(
        np.eye(2)
        + np.matrix(
            [
            [0, dt],
            [-(g/L)*np.cos(X[0])*dt, -(b/(m*L**2))*dt]
            ]
        )
    )
def return_B(X,U,dt):
    return(
        np.matrix(
            [
            [0],
            [dt/(m*L**2)]
            ]
        )
    )
def return_linearized_dynamics_matrices(X,U,dt):
    """
    Takes in the input U and the the corresponding output X, as well as dt and returns two lists that contain the linearized dynamic matrices for each timestep for range(len(Time)-1).
    """
    # returns a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
    Phi = list(
            map(
                lambda X,U: return_Phi(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
    B = list(
            map(
                lambda X,U: return_B(X,U,dt),
                X[:,:-1].T,
                U
            )
        )
    return(Phi,B)

# for case where time to target is penalized. Hoping to create steady state.
def return_l_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[(1/2)*U**2*dt]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[k1*(1/2)*(X[0]-TargetAngle)**2*dt]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt:\
                    np.matrix([[k2*(1/2)*(X[1]-TargetAngularVelocity)**2*dt]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt)
    return(result)
def return_lx_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    result = lambda X,U,dt: np.matrix([[0],[0]])
    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[0],[0]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0],[0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[k1*(X[0]-TargetAngle)*dt],[0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0],[0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0],[k2*(X[1]-TargetAngularVelocity)*dt]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0],[0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt)
    return(result)
def return_lu_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[U*dt]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt)
    return(result)
def return_lxu_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[0],[0]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0],[0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[0],[0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0],[0]])
    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0],[0]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0],[0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt)
    return(result)
def return_lux_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[0,0]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0,0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[0,0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0,0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0,0]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0,0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt)
    return(result)
def return_luu_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[dt]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0]])
    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt)
    return(result)
def return_lxx_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[0,0],[0,0]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0,0],[0,0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[k1*1*dt,0],[0,0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0,0],[0,0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0,0],[0,k2*1*dt]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0,0],[0,0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt)
    return(result)

def return_quadratic_cost_function_expansion_variables(
        X,U,dt,
        RunningCost="Minimize Input Energy"):
    """
    Takes in the input U and the the corresponding output X, as well as dt and returns lists that contain the coefficient matrices for the quadratic expansion of the cost function (l(x,u)) for each timestep for range(len(Time)-1).
    """

    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    # returns a list of length len(Time)-1, each element with shape (1,1), where n is the number of states.
    l_func = return_l_func(RunningCost=RunningCost)
    l = list(
            map(
                lambda X,U: l_func(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,1), where n is the number of states.
    lx_func = return_lx_func(RunningCost=RunningCost)
    lx = list(
            map(
                lambda X,U: lx_func(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,1), where n is the number of states.
    lu_func = return_lu_func(RunningCost=RunningCost)
    lu = list(
            map(
                lambda X,U: lu_func(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,n), where m is the number of inputs and n is the number of states.
    lux_func = return_lux_func(RunningCost=RunningCost)
    lux = list(
            map(
                lambda X,U: lux_func(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
    lxu_func = return_lxu_func(RunningCost=RunningCost)
    lxu = list(
            map(
                lambda X,U: lxu_func(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,m), where m is the number of inputs.
    luu_func = return_luu_func(RunningCost=RunningCost)
    luu = list(
            map(
                lambda X,U: luu_func(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
    lxx_func = return_lxx_func(RunningCost=RunningCost)
    lxx = list(
            map(
                lambda X,U: lxx_func(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    return(l,lx,lu,lux,lxu,luu,lxx)

def return_terminal_cost_func(
        TerminalCost='Minimize final angle',
        ReturnGradientAndHessian=False):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    Cost should be either 'Minimize final angle from target angle' (Default), 'Minimize final angular velocity from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(TerminalCost)==str:
        assert TerminalCost in ['Minimize final angle from target angle',
                'Minimize final angular velocity from target angular velocity'],\
            "TerminalCost must be either 'Minimize final angle from target angle' (Default), 'Minimize final angular velocity from target angular velocity'."
    else:
        assert type(TerminalCost)==list, "TerminalCost must be a list of cost types."
        for el in TerminalCost:
            assert type(el)==str, "Each element of TerminalCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize final angle from target angle',
                    'Minimize final angular velocity from target angular velocity'],\
                "Each element of TerminalCost must be either 'Minimize final angle from target angle' (Default), 'Minimize final angular velocity from target angular velocity'. '" + el + "' not accepted."

    assert type(ReturnGradientAndHessian)==bool, \
            "ReturnGradientAndHessian must be a bool."

    if "Minimize final angle from target angle" in TerminalCost:
        result1 = lambda X,U,dt: t1*(1/2)*(X[0,-1]-TargetAngle)**2
        dx_result1 = lambda X,U,dt: np.matrix([[t1*(X[0,-1]-TargetAngle)],[0]])
        dxx_result1 = lambda X,U,dt: np.matrix([[t1*1,0],[0,0]])
    else:
        result1 = lambda X,U,dt: 0
        dx_result1 = lambda X,U,dt: np.matrix([[0],[0]])
        dxx_result1 = lambda X,U,dt: np.matrix([[0,0],[0,0]])

    if "Minimize final angular velocity from target angular velocity" in TerminalCost:
        result2 = lambda X,U,dt: t2*(1/2)*(X[1,-1]-TargetAngularVelocity)**2
        dx_result2 = lambda X,U,dt:\
                np.matrix([[0],[t2*(X[1,-1]-TargetAngularVelocity)]])
        dxx_result2 = lambda X,U,dt: np.matrix([[0,0],[0,t2*1]])
    else:
        result2 = lambda X,U,dt: 0
        dx_result2 = lambda X,U,dt: np.matrix([[0],[0]])
        dxx_result2 = lambda X,U,dt: np.matrix([[0,0],[0,0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt)
    if ReturnGradientAndHessian==False:
        return(result)
    else:
        dx_result = lambda X,U,dt: dx_result1(X,U,dt) \
                            + dx_result2(X,U,dt)
        dxx_result = lambda X,U,dt: dxx_result1(X,U,dt) \
                            + dxx_result2(X,U,dt)
        return(result,dx_result,dxx_result)

def return_running_cost_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.trapz((1/2)*U**2,dx=dt)
    else:
        result1 = lambda X,U,dt: 0

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt:\
                    np.trapz(k1*(1/2)*(X[0,:]-TargetAngle)**2,dx=dt)
    else:
        result2 = lambda X,U,dt: 0

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt:\
                    np.trapz(k2*(1/2)*(X[1,:]-TargetAngularVelocity)**2,dx=dt)
    else:
        result3 = lambda X,U,dt: 0

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt)
    return(result)

def return_empty_lists_for_quadratic_expansion_of_Q(length):
    Qu = [None]*length
    Qx = [None]*length
    Qux = [None]*length
    Qxu = [None]*length
    Quu = [None]*length
    Qxx = [None]*length
    return(Qu,Qx,Qux,Qxu,Quu,Qxx)

def simple_pendulum_ddp(**kwargs):

    RunningCost = kwargs.get("RunningCost","Minimize Input Energy")
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'."
    else:
        assert type(RunningCost)==list, \
                "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', or 'Minimize time away from target angular velocity'. '" + el + "' not accepted."
    running_cost_func = return_running_cost_func(RunningCost=RunningCost)

    TerminalCost = kwargs.get("TerminalCost","Minimize final angle from target angle")
    if type(TerminalCost)==str:
        assert TerminalCost in ['Minimize final angle from target angle',
                'Minimize final angular velocity from target angular velocity'],\
            "TerminalCost must be either 'Minimize final angle from target angle' (Default), 'Minimize final angular velocity from target angular velocity'."
    else:
        assert type(TerminalCost)==list, "TerminalCost must be a list of cost types."
        for el in TerminalCost:
            assert type(el)==str, "Each element of TerminalCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize final angle from target angle',
                    'Minimize final angular velocity from target angular velocity'],\
                "Each element of TerminalCost must be either 'Minimize final angle from target angle' (Default), 'Minimize final angular velocity from target angular velocity'. '" + el + "' not accepted."
    terminal_cost_func, terminal_cost_func_grad, terminal_cost_func_hess = \
            return_terminal_cost_func(
                    TerminalCost=TerminalCost,
                    ReturnGradientAndHessian=True
                    )

    ICs = kwargs.get("ICs",[0,0]) # in degrees
    assert type(ICs)==list and len(ICs)==2, "ICs must be a list of length 2."
    LocationStrings = ["1st", "2nd"]
    for i in range(2):
        assert str(type(ICs[i])) in [
                "<class 'numpy.float'>",
                "<class 'int'>",
                "<class 'float'>"],\
            "ICs must be numbers. Check the " + LocationString[i] + " element of IC"

    dt = kwargs.get("dt",0.01)
    assert str(type(dt)) in [
            "<class 'numpy.float'>",
            "<class 'int'>",
            "<class 'float'>"],\
        "dt must be a number."

    N_seconds = kwargs.get("N_seconds",10)
    assert str(type(N_seconds)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>"],\
        "N_seconds must be a number."

    N_iterations = kwargs.get("N_iterations",10)
    assert str(type(N_iterations)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>"],\
        "N_iterations must be a number."

    TotalX = []
    TotalU = []

    Time = np.arange(0,N_seconds + dt,dt)

    U = kwargs.get("U",2*np.ones(len(Time)-1)) # initial input guess
    assert len(U)==len(Time)-1, "U must be an array with length len(Time)-1."

    TrialCosts = []
    IterationNumber = 1
    while IterationNumber <= N_iterations:
        print(
            "Iteration Number : "
            + str(IterationNumber)
            + "/"
            + str(N_iterations)
        )
        X = forward_integrate_dynamics(
                ICs,
                U=U,
                dt=dt,
                N_seconds=N_seconds,
                ReturnX=True)
        TotalX.append(X)
        TotalU.append(U)

        TrialCosts.append(
               terminal_cost_func(X,U,dt)
               + running_cost_func(X,U,dt)
               )

        Phi,B = return_linearized_dynamics_matrices(X,U,dt)
        l,lx,lu,lux,lxu,luu,lxx = \
                return_quadratic_cost_function_expansion_variables(X,U,dt,RunningCost=RunningCost)

        # initializing the lists (to be built backwards)
        #   # Only concerned with reaching the target angle, dont care about velocity.
        # V = [np.matrix([[(1/2)*(X[0,-1] - TargetAngle)**2]])] # could add (1/2)*X[1,-1]**2
        # Vx = [np.matrix([[(X[0,-1]-TargetAngle)],[0]])] # could include X[1,-1] for Vx[0,0]
        # Vxx = [np.matrix([[1,0],[0,0]])] # could include 1 for the Vxx[0,1]

            # Placing a cost on the target velocity as well.
        V = [np.matrix([[terminal_cost_func(X,U,dt)]])]
        Vx = [terminal_cost_func_grad(X,U,dt)]
        Vxx = [terminal_cost_func_hess(X,U,dt)]

        Qu,Qx,Qux,Qxu,Quu,Qxx = \
                return_empty_lists_for_quadratic_expansion_of_Q(len(Time)-1)
        # statusbar1 = dsb(0,len(Time)-1,title="Backward Pass")
        for i in range(len(Time)-1):
            Qx[-(i+1)] = lx[-(i+1)] + Phi[-(i+1)].T*Vx[-1]
            Qu[-(i+1)] = lu[-(i+1)] + B[-(i+1)].T*Vx[-1]
            Qux[-(i+1)] = lux[-(i+1)] + B[-(i+1)].T*Vxx[-1]*Phi[-(i+1)]
            Qxu[-(i+1)] = lxu[-(i+1)] + Phi[-(i+1)].T*Vxx[-1]*B[-(i+1)]
            Quu[-(i+1)] = luu[-(i+1)] + B[-(i+1)].T*Vxx[-1]*B[-(i+1)]
            Qxx[-(i+1)] = lxx[-(i+1)] + Phi[-(i+1)].T*Vxx[-1]*Phi[-(i+1)]

            # V.append(
            #         l[-(i+1)]
            #         + V[-1]
            #         - (1/2)*Qu[-(i+1)].T*(Quu[-(i+1)]**(-1))*Qu[-(i+1)]
            #     )
            V.append(
                     V[-1]
                    - Qu[-(i+1)].T*(Quu[-(i+1)]**(-1))*Qu[-(i+1)]
                )

            Vx.append(
                    Qx[-(i+1)]
                    - Qxu[-(i+1)]*(Quu[-(i+1)]**(-1))*Qu[-(i+1)]
                )

            Vxx.append(
                    Qxx[-(i+1)]
                    - Qxu[-(i+1)]*(Quu[-(i+1)]**(-1))*Qux[-(i+1)]
                )
            # statusbar1.update(i)

        V = list(reversed(V))
        Vx = list(reversed(Vx))
        Vxx = list(reversed(Vxx))

        dx = [None]*len(Time)
        du = [None]*(len(Time)-1)

        dx[0] = np.matrix([[0],[0]])

        # statusbar2 = dsb(0,len(Time)-1,title="Forward Pass")
        for i in range(len(Time)-1):
            du[i] = -Quu[i]**(-1)*(Qux[i]*dx[i] + Qu[i])
            dx[i+1] = Phi[i]*dx[i] + B[i]*du[i]
            # statusbar2.update(i)

        U = np.array(U + np.concatenate(du,axis=0).T)[0]

        IterationNumber+=1

    X = forward_integrate_dynamics(
            [0,0],
            U=U,
            dt=dt,
            N_seconds=N_seconds,
            ReturnX=True)
    TotalX.append(X)
    TotalU.append(U)
    TrialCosts.append(
           terminal_cost_func(X,U,dt)
           + running_cost_func(X,U,dt)
           )

    animate_trajectory(Time,X,U)

    plt.figure()
    plt.plot(
        list(range(N_iterations+1)),
        TrialCosts,
        marker = 'o')
    plt.show()

    return(Time,TotalX,TotalU,TrialCosts)
