import numpy as np
from dynamics import *
from animate import *

def forward_integrate_dynamics(ICs,U=None,**kwargs):
    """
    ICs must be a list of floats and/or ints of length 4. If ReturnX is True, the this will return an array of shape (4,len(Time)).

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

    X1: cart X-position
    X2: angle of pendulum
    X3: cart X-velocity
    X4: angular velocity of pendulum

    """
    assert np.shape(ICs)==(4,), "ICs must be a numpy array of shape (4,)."
    LocationStrings = ["1st", "2nd", "3rd", "4th"]
    for i in range(4):
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
    X = np.zeros((4,Horizon))
    if U is None:
        U = np.zeros(Horizon-1)
    else:
        assert len(U)==Horizon-1, "U must have length = (Horizon-1)."
        # if not (len(U)==len(Time)-1):
        #     import ipdb; ipdb.set_trace()

    # ICs

    if UsingDegrees:
        X[0,0] = ICs[0]
        X[1,0] = ICs[1]*(np.pi/180)
        X[2,0] = ICs[2]
        X[3,0] = ICs[3]*(np.pi/180)
    else:
        X[0,0] = ICs[0]
        X[1,0] = ICs[1]
        X[2,0] = ICs[2]
        X[3,0] = ICs[3]

    for i in range(Horizon-1):
        X[0,i+1] = X[0,i] + F1(X[:,i],U[i])*dt
        X[1,i+1] = X[1,i] + F2(X[:,i],U[i])*dt
        X[2,i+1] = X[2,i] + F3(X[:,i],U[i])*dt
        X[3,i+1] = X[3,i] + F4(X[:,i],U[i])*dt

    if AnimateStates==False and PlotStates==False:
        return(X)
    else:
        if AnimateStates:
            animate_trajectory(Time,X,U)
        if PlotStates:
            plt.figure(figsize=(15,10))

            ax1 = plt.subplot(323)
            ax1.plot(Time,180*X[1,:]/np.pi,'gr')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Pendulum Angle (deg)')
            if max(abs(180*X[1,:]/np.pi - 180*X[1,0]/np.pi))<1e-7:
                ax1.set_ylim([180*X[1,0]/np.pi - 5,180*X[1,0]/np.pi + 5])

            ax2 = plt.subplot(324)
            ax2.plot(Time,X[0,:],'r')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Cart Position (m)')
            if max(abs(X[0,:] - X[0,0]))<1e-7:
                ax2.set_ylim([X[0,0] - 5,X[0,0] + 5])

            ax3 = plt.subplot(325)
            ax3.plot(Time,180*X[3,:]/np.pi,'gr--')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Pendulum Angular \n Velocity (deg/s)')
            if max(abs(180*X[3,:]/np.pi-180*X[3,0]/np.pi))<1e-7:
                ax3.set_ylim([180*X[3,0]/np.pi-1,180*X[3,0]/np.pi+1])

            ax4 = plt.subplot(326)
            ax4.plot(Time,X[2,:],'r--')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Cart Velocity (m/s)')
            if max(abs(X[2,:] - X[2,0]))<1e-7:
                ax4.set_ylim([X[2,0] - 5,X[2,0] + 5])

            # ax5 = plt.subplot2grid((3,2),(0,0),colspan=2)
            ax5 = plt.subplot(322)
            ax5.plot(Time[:-1],U[:],'b')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Input Force (N)')
            if max(abs(U[:] - U[0]))<1e-7:
                ax5.set_ylim([U[0] - 5,U[0] + 5])

            # ax0 = plt.subplot2grid((3,4),(0,0),colspan=2) # animation
            ax0 = plt.subplot(321)

            Pendulum_Width = 0.01*L
            Pendulum_Length = 2*L
            Cart_Width = 4*L
            Cart_Height = 2*L
            Wheel_Radius = 0.125*Cart_Width

            Cart = plt.Rectangle(
                        (-Cart_Width/2,-Cart_Height/2),
                        Cart_Width,
                        Cart_Height,
                        Color='#4682b4')
            ax0.add_patch(Cart)

            FrontWheel = plt.Circle(
                        (Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                        Wheel_Radius,
                        Color='k')
            ax0.add_patch(FrontWheel)
            FrontWheel_Rivet = plt.Circle(
                        (Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                        0.2*Wheel_Radius,
                        Color='0.70')
            ax0.add_patch(FrontWheel_Rivet)

            BackWheel = plt.Circle(
                        (-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                        Wheel_Radius,
                        Color='k')
            ax0.add_patch(BackWheel)
            BackWheel_Rivet = plt.Circle(
                        (-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                        0.2*Wheel_Radius,
                        Color='0.70')
            ax0.add_patch(BackWheel_Rivet)

            Pendulum, = ax0.plot(
                            [
                                0,
                                Pendulum_Length*np.sin((30*np.pi/180))
                            ],
                            [
                                Cart_Height/2 + Pendulum_Width/2,
                                Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos((30*np.pi/180))
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
                                Cart_Height/2 + Pendulum_Width/2,
                                Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length
                            ],
                            Color='k',
                            lw = 1,
                            linestyle='--'
                            )

            Angle_indicator, = ax0.plot(
                            Pendulum_Length*np.sin(
                                np.linspace(0.05*(30*np.pi/180),0.95*(30*np.pi/180),20)
                                ),
                            Cart_Height/2
                            + Pendulum_Width/2
                            + Pendulum_Length*np.cos(
                                np.linspace(0.05*(30*np.pi/180),0.95*(30*np.pi/180),20)
                                ),
                            Color='gr',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            k = 0.15*Pendulum_Length
            Angle_indicator_arrow, = ax0.plot(
                            Pendulum_Length*np.sin(0.95*(30*np.pi/180))
                            + [
                                -k*np.cos(0.95*(30*np.pi/180) + (30*np.pi/180)),
                                0,
                                -k*np.cos(0.95*(30*np.pi/180) - (30*np.pi/180))
                            ],
                            Cart_Height/2
                            + Pendulum_Width/2
                            + Pendulum_Length*np.cos(0.95*(30*np.pi/180))
                            + [
                                k*np.sin(0.95*(30*np.pi/180) + (30*np.pi/180)),
                                0,
                                k*np.sin(0.95*(30*np.pi/180) - (30*np.pi/180))
                            ],
                            Color='gr',
                            lw = 2,
                            solid_capstyle='round'
                            )
            Angle_damping_indicator, = ax0.plot(
                            0.5*Pendulum_Length*np.sin(
                                    np.linspace(
                                        1.05*(30*np.pi/180),
                                        1.05*(30*np.pi/180)+(45*np.pi/180),
                                        20
                                    )
                                ),
                            Cart_Height/2
                            + Pendulum_Width/2
                            + 0.5*Pendulum_Length*np.cos(
                                    np.linspace(
                                        1.05*(30*np.pi/180),
                                        1.05*(30*np.pi/180)+(45*np.pi/180),
                                        20
                                    )
                                ),
                            Color='#ffa500',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            Angle_damping_indicator_arrow, = ax0.plot(
                            0.5*Pendulum_Length*np.sin(1.05*(30*np.pi/180))
                            + [
                                -k*np.cos(1.05*(30*np.pi/180) + (30*np.pi/180) + np.pi),
                                0,
                                -k*np.cos(1.05*(30*np.pi/180) - (30*np.pi/180) + np.pi)
                            ],
                            Cart_Height/2
                            + Pendulum_Width/2
                            + 0.5*Pendulum_Length*np.cos(1.05*(30*np.pi/180))
                            + [
                                k*np.sin(1.05*(30*np.pi/180) + (30*np.pi/180) + np.pi),
                                0,
                                k*np.sin(1.05*(30*np.pi/180) - (30*np.pi/180) + np.pi)
                            ],
                            Color='#ffa500',
                            lw = 2,
                            solid_capstyle='round'
                            )

            Position_damping_indicator, = ax0.plot(
                            [0.55*Cart_Width,0.90*Cart_Width],
                            [0]*2,
                            Color='#800080',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            Position_damping_indicator_arrow, = ax0.plot(
                            0.55*Cart_Width
                            - np.array([
                                -k*np.cos((30*np.pi/180)),
                                0,
                                -k*np.cos(-(30*np.pi/180))
                            ]),
                            - np.array([
                                k*np.sin((30*np.pi/180)),
                                0,
                                k*np.sin(-(30*np.pi/180))
                            ]),
                            Color='#800080',
                            lw = 2,
                            solid_capstyle='round'
                            )


            Pendulum_Attachment = plt.Circle((0,Cart_Height/2),100*Pendulum_Width/2,Color='#4682b4')
            ax0.add_patch(Pendulum_Attachment)

            Pendulum_Rivet, = ax0.plot(
                [0],
                [Cart_Height/2 + Pendulum_Width/2],
                c='k',
                marker='o',
                lw=2
                )

            Ground = plt.Rectangle(
                        (-10,-1.50*(Cart_Height/2 + Wheel_Radius*2)),
                        20,
                        0.50*(Cart_Height/2 + Wheel_Radius*2),
                        Color='0.70')
            ax0.add_patch(Ground)

            Position_indicator, = ax0.plot(
                            [-k,5*k],
                            [-1.25*(Cart_Height/2 + Wheel_Radius*2)]*2,
                            Color='r',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            Position_indicator_cross, = ax0.plot(
                            [0,0],
                            [
                                -1.25*(Cart_Height/2 + Wheel_Radius*2)+k,
                                -1.25*(Cart_Height/2 + Wheel_Radius*2)-k
                            ],
                            Color='r',
                            lw = 2,
                            solid_capstyle = 'round'
                            )

            Position_indicator_arrow, = ax0.plot(
                            5*k
                            + np.array([
                                -k*np.cos((30*np.pi/180)),
                                0,
                                -k*np.cos(-(30*np.pi/180))
                            ]),
                            -1.25*(Cart_Height/2 + Wheel_Radius*2)
                            + np.array([
                                k*np.sin((30*np.pi/180)),
                                0,
                                k*np.sin(-(30*np.pi/180))
                            ]),
                            Color='r',
                            lw = 2,
                            solid_capstyle='round'
                            )

            Force_indicator, = ax0.plot(
                            [-0.90*Cart_Width,-0.55*Cart_Width],
                            [0]*2,
                            Color='b',
                            lw = 2,
                            solid_capstyle = 'round'
                            )
            Force_indicator_arrow, = ax0.plot(
                            -0.55*Cart_Width
                            + np.array([
                                -k*np.cos((30*np.pi/180)),
                                0,
                                -k*np.cos(-(30*np.pi/180))
                            ]),
                            np.array([
                                k*np.sin((30*np.pi/180)),
                                0,
                                k*np.sin(-(30*np.pi/180))
                            ]),
                            Color='b',
                            lw = 2,
                            solid_capstyle='round'
                            )

            ax0.get_xaxis().set_ticks([])
            ax0.get_yaxis().set_ticks([])
            ax0.set_frame_on(True)
            ax0.set_xlim([-1*Cart_Width,1*Cart_Width])
            ax0.set_ylim(
                [
                    -1.50*(Cart_Height/2 + Wheel_Radius*2),
                    1.25*(Cart_Height/2+Pendulum_Width/2+Pendulum_Length+Pendulum_Width/2)
                ]
            )

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # ax0.text(0.05, 0.95, r"$b_1$ = " + str(b1) + "\n" + r"$b_2$ = " + str(b2), transform=ax0.transAxes, fontsize=14,
            # verticalalignment='top', bbox=props)
            ax0.legend(
                (Position_damping_indicator,Angle_damping_indicator),
                (r"$b_1$ = " + str(b1), r"$b_2$ = " + str(b2)),
                loc='upper left',
                facecolor='wheat',
                framealpha=0.5,
                title="Damping")
            # ax0.set_ylabel(r"$b_1$ = " + str(b1) + r"\n $b_2$ = " + str(b2))
            ax0.set_aspect('equal')

            plt.show()
