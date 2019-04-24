import numpy as np
from dynamics import *
from animate import *

def forward_integrate_dynamics(ICs,U=None,**kwargs):
    """
    ICs must be a list of floats and/or ints of length 2. If ReturnX is True, the this will return an array of shape (2,len(Time)).

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

    X1: angle of pendulum
    X2:  angular velocity of pendulum

    """
    assert np.shape(ICs)==(2,), "ICs must be a numpy array of shape (2,)."
    LocationStrings = ["1st", "2nd"]
    for i in range(2):
        assert str(type(ICs[i])) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
            "ICs must be numbers. Check the " + LocationStrings[i] + " element of IC"

    dt = kwargs.get("dt",0.01)
    assert str(type(dt)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
        "dt must be a number."

    Horizon = kwargs.get("Horizon",300)
    assert str(type(Horizon)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
        "Horizon must be a number."

    UsingDegrees = kwargs.get("UsingDegrees",False)
    assert type(UsingDegrees)==bool, "UsingDegrees must be either True or False (Default)."

    AnimateStates = kwargs.get("AnimateStates",False)
    assert type(AnimateStates)==bool, "AnimateStates must be either True or False (Default)."

    PlotStates = kwargs.get("PlotStates",False)
    assert type(PlotStates)==bool, "PlotStates must be either True or False (Default)."

    Time = np.arange(0,Horizon*dt,dt)
    X = np.zeros((2,Horizon))
    if U is None:
        U = np.zeros((2,Horizon-1))
    else:
        assert np.shape(U)==(2,Horizon-1), "U must have shape = (2,Horizon-1)."

    # ICs
    if UsingDegrees:
        X[0,0] = ICs[0]*(np.pi/180)
        X[1,0] = ICs[1]*(np.pi/180)
    else:
        X[0,0] = ICs[0]
        X[1,0] = ICs[1]

    for i in range(Horizon-1):
        X[0,i+1] = X[0,i] + F1(X[:,i],U[:,i])*dt
        X[1,i+1] = X[1,i] + F2(X[:,i],U[:,i])*dt


    if AnimateStates==False and PlotStates==False:
        return(X)
    else:
        if AnimateStates:
            animate_trajectory(Time,X,U)
        if PlotStates:
            plt.figure(figsize=(15,10))

            # ax1 = plt.subplot2grid((3,2),(0,0),colspan=2)
            ax1 = plt.subplot(222)
            ax1.plot(Time[:-1],U[0,:],'r')
            ax1.plot(Time[:-1],U[1,:],'g')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Tendon Tension (N)')
            if max(abs(U[0,:] - U[0,0]))<1e-7 and max(abs(U[1,:] - U[1,0]))<1e-7:
                ax1.set_ylim([min(U[:,0]) - 5,max(U[:,0]) + 5])

            ax2 = plt.subplot(223)
            ax2.plot(Time,180*X[0,:]/np.pi,'b')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Angle (deg)')
            if max(abs(180*X[0,:]/np.pi - 180*X[0,0]/np.pi))<1e-7:
                ax2.set_ylim([180*X[0,0]/np.pi - 5,180*X[0,0]/np.pi + 5])

            ax3 = plt.subplot(224)
            ax3.plot(Time,180*X[1,:]/np.pi,'b--')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Angular Velocity (deg/s)')
            if max(abs(180*X[1,:]/np.pi-180*X[1,0]/np.pi))<1e-7:
                ax3.set_ylim([180*X[1,0]/np.pi-1,180*X[1,0]/np.pi+1])

            ax0 = plt.subplot(221)
            Pendulum_Width = 0.01*L1
            Pendulum_Length = L1

            Ground = plt.Rectangle(
                        (-52*Pendulum_Width/4,-Pendulum_Length/4),
                        52*Pendulum_Width/4,
                        Pendulum_Length/2,
                        Color='#4682b4')
            ax0.add_patch(Ground)


            Pendulum, = ax0.plot(
                            [
                                0,
                                Pendulum_Length*np.sin((30*np.pi/180))
                            ],
                            [
                                0,
                                -Pendulum_Length*np.cos((30*np.pi/180))
                            ],
                            Color='0.50',
                            lw = 10,
                            solid_capstyle='round'
                            )

            Pendulum_neutral, = ax0.plot(
                            [
                                0,
                                0
                            ],
                            [
                                0,
                                -Pendulum_Length
                            ],
                            Color='k',
                            lw = 1,
                            linestyle='--'
                            )

            Angle_indicator, = ax0.plot(
                            Pendulum_Length*np.sin(
                                np.linspace(0.05*(30*np.pi/180),0.95*(30*np.pi/180),20)
                                ),
                            -Pendulum_Length*np.cos(
                                np.linspace(0.05*(30*np.pi/180),0.95*(30*np.pi/180),20)
                                ),
                            Color='b',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            k = 0.075*Pendulum_Length
            Angle_indicator_arrow, = ax0.plot(
                            Pendulum_Length*np.sin(0.95*(30*np.pi/180))
                            + [
                                -k*np.sin((120*np.pi/180) - 0.95*(30*np.pi/180)),
                                0,
                                -k*np.sin((60*np.pi/180) - 0.95*(30*np.pi/180))
                            ],
                            -Pendulum_Length*np.cos(0.95*(30*np.pi/180))
                            + [
                                -k*np.cos((120*np.pi/180) - 0.95*(30*np.pi/180)),
                                0,
                                -k*np.cos((60*np.pi/180) - 0.95*(30*np.pi/180))
                            ],
                            Color='b',
                            lw = 2,
                            solid_capstyle='round'
                            )
            Angle_damping_indicator, = ax0.plot(
                            0.50*Pendulum_Length*np.sin(
                                    np.linspace(
                                        0.45*(30*np.pi/180),
                                        1.55*(30*np.pi/180),
                                        20
                                    )
                                ),
                            -0.50*Pendulum_Length*np.cos(
                                    np.linspace(
                                        0.45*(30*np.pi/180),
                                        1.55*(30*np.pi/180),
                                        20
                                    )
                                ),
                            Color='#ffa500',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            Angle_damping_indicator_arrow, = ax0.plot(
                            0.50*Pendulum_Length*np.sin(0.45*(30*np.pi/180))
                            + [
                                k*np.sin(0.45*(30*np.pi/180) + (60*np.pi/180)),
                                0,
                                k*np.sin(0.45*(30*np.pi/180) + (120*np.pi/180))
                            ],
                            -0.50*Pendulum_Length*np.cos(0.45*(30*np.pi/180))
                            + [
                                -k*np.cos(0.45*(30*np.pi/180) + (60*np.pi/180)),
                                0,
                                -k*np.cos(0.45*(30*np.pi/180) + (120*np.pi/180))
                            ],
                            Color='#ffa500',
                            lw = 2,
                            solid_capstyle='round'
                            )

            tau1_indicator, = ax0.plot(
                            0.75*Pendulum_Length*np.sin(
                                    np.linspace(
                                        1.05*(30*np.pi/180),
                                        1.05*(30*np.pi/180)+(45*np.pi/180),
                                        20
                                    )
                                ),
                            -0.75*Pendulum_Length*np.cos(
                                    np.linspace(
                                        1.05*(30*np.pi/180),
                                        1.05*(30*np.pi/180)+(45*np.pi/180),
                                        20
                                    )
                                ),
                            Color='r',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            tau1_indicator_arrow, = ax0.plot(
                            0.75*Pendulum_Length*np.sin(1.05*(30*np.pi/180)+(45*np.pi/180))
                            + [
                                -k*np.sin((120*np.pi/180) - 1.05*(30*np.pi/180)-(45*np.pi/180)),
                                0,
                                -k*np.sin((60*np.pi/180) - 1.05*(30*np.pi/180)-(45*np.pi/180))
                            ],
                            -0.75*Pendulum_Length*np.cos(1.05*(30*np.pi/180)+(45*np.pi/180))
                            + [
                                -k*np.cos((120*np.pi/180) - 1.05*(30*np.pi/180)-(45*np.pi/180)),
                                0,
                                -k*np.cos((60*np.pi/180) - 1.05*(30*np.pi/180)-(45*np.pi/180))
                            ],
                            Color='r',
                            lw = 2,
                            solid_capstyle='round'
                            )

            tau2_indicator, = ax0.plot(
                            0.75*Pendulum_Length*np.sin(
                                    np.linspace(
                                        0.95*(30*np.pi/180)-(45*np.pi/180),
                                        0.95*(30*np.pi/180),
                                        20
                                    )
                                ),
                            -0.75*Pendulum_Length*np.cos(
                                    np.linspace(
                                        0.95*(30*np.pi/180)-(45*np.pi/180),
                                        0.95*(30*np.pi/180),
                                        20
                                    )
                                ),
                            Color='g',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            tau2_indicator_arrow, = ax0.plot(
                            0.75*Pendulum_Length*np.sin(0.95*(30*np.pi/180)-(45*np.pi/180))
                            + [
                                k*np.sin((15*np.pi/180) + 0.95*(30*np.pi/180)),
                                0,
                                k*np.sin((75*np.pi/180) + 0.95*(30*np.pi/180))
                            ],
                            -0.75*Pendulum_Length*np.cos(0.95*(30*np.pi/180)-(45*np.pi/180))
                            + [
                                -k*np.cos((15*np.pi/180) + 0.95*(30*np.pi/180)),
                                0,
                                -k*np.cos((75*np.pi/180) + 0.95*(30*np.pi/180))
                            ],
                            Color='g',
                            lw = 2,
                            solid_capstyle='round'
                            )


            Pendulum_Attachment = plt.Circle((0,0),50*Pendulum_Width/4,Color='#4682b4')
            ax0.add_patch(Pendulum_Attachment)

            Pendulum_Rivet, = ax0.plot(
                [0],
                [0],
                c='k',
                marker='o',
                lw=2
                )

            ax0.get_xaxis().set_ticks([])
            ax0.get_yaxis().set_ticks([])
            ax0.set_frame_on(True)
            ax0.set_xlim([-0.60*Pendulum_Length,1.00*Pendulum_Length])
            ax0.set_ylim([-1.10*Pendulum_Length,0.30*Pendulum_Length])

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # ax0.text(0.05, 0.95, r"$b_1$ = " + str(b1) + "\n" + r"$b_2$ = " + str(b2), transform=ax0.transAxes, fontsize=14,
            # verticalalignment='top', bbox=props)
            ax0.legend(
                (Angle_damping_indicator,tau1_indicator,tau2_indicator),
                (r"$b_1\dot{\theta}$", r"$R_1(\theta)u_1$", r"$R_2(\theta)u_2$"),
                loc='upper left',
                facecolor='wheat',
                framealpha=0.5,
                title="Torques")
            ax0.set_aspect('equal')

            plt.show()
