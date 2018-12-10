import numpy as np
import matplotlib.pyplot as plt
from danpy.sb import *
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy import signal

"""
Notes:

X1: position of the cart
X2: angle of pendulum
X3: velocity of the cart
X4: angular velocity of pendulum

dt should be set smaller for this problem as integration error can effect the behavior. See what happens for forward_integrate_dynamics([170,0,0,0],dt=0.01). Try dt = 0.0001

"""
def dx1_dt(X,U):
    return(X[2])
def dx2_dt(X,U):
    return(X[3])
def dx3_dt(X,U):
    return(
        (
            m2*L*np.sin(X[1])*(X[3]**2)
            - b1*X[2]
            - m2*g*np.sin(2*X[1])/2
            + b2*np.cos(X[1])*X[3]/L
            + U
        )
        /
        (m1 + m2*(np.sin(X[1])**2))
    )
def dx4_dt(X,U):
    return(
        (
            -m2*np.sin(2*X[1])*(X[3]**2)/2
            + b1*np.cos(X[1])*X[2]/L
            + (m1+m2)*g*np.sin(X[1])/L
            - (m1+m2)*b2*X[3]/(m2*(L**2))
            - np.cos(X[1])*U/L
        )
        /
        (m1 + m2*(np.sin(X[1])**2))
    )

def forward_integrate_dynamics(
        ICs,
        UsingDegrees=True,
        Animate=False,
        U=None,
        ReturnX=False,
        **kwargs):
    """
    ICs must be a list of floats and/or ints of length 4. If ReturnX is True, the this will return an array of shape (4,len(Time)).

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **kwargs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    UsingDegrees must be a bool. Default is True. If True, then the ICs for pendulum angle and angular velocity can be given in degrees and degrees per second, respectively.

    Animate must be a bool. Default is False. If True, the program will run animate_trajectory().

    dt must be a number. Default is 0.01. Used with N_seconds to define the time array (Time).

    N_seconds must be a number. Default is 10. Used with dt to define the time array (Time).

    U can either be None (default) or can be an array with lenth (len(Time)-1). If None, then U will be chosen to be np.zeros(len(Time)-1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Notes:

    X1: angle of pendulum
    X2: angular velocity of pendulum

    """
    assert type(ICs)==list and len(ICs)==4, "ICs must be a list of length 2."
    LocationStrings = ["1st", "2nd", "3rd", "4th"]
    for i in range(4):
        assert str(type(ICs[i])) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
            "ICs must be numbers. Check the " + LocationStrings[i] + " element of IC"

    assert type(UsingDegrees)==bool, "UsingDegrees must be either True or False."

    assert type(Animate)==bool, "Animate must be either True or False."

    dt = kwargs.get("dt",0.01)
    assert str(type(dt)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
        "dt must be a number."

    N_seconds = kwargs.get("N_seconds",10)
    assert str(type(N_seconds)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
        "N_seconds must be a number."

    Time = np.arange(0,N_seconds+dt,dt)
    X = np.zeros((4,len(Time)))
    if U is None:
        U = np.zeros(len(Time)-1)
    else:
        assert len(U)==len(Time)-1, "U must have length = (len(Time)-1)."

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

    for i in range(len(Time)-1):
        X[0,i+1] = X[0,i] + dx1_dt(X[:,i],U[i])*dt
        X[1,i+1] = X[1,i] + dx2_dt(X[:,i],U[i])*dt
        X[2,i+1] = X[2,i] + dx3_dt(X[:,i],U[i])*dt
        X[3,i+1] = X[3,i] + dx4_dt(X[:,i],U[i])*dt


    if ReturnX==True:
        return(X)
    else:
        if Animate:
            animate_trajectory(Time,X,U)
        else:
            plt.figure(figsize=(15,10))

            ax1 = plt.subplot(323)
            ax1.plot(Time,180*X[1,:]/np.pi,'g')
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
            ax3.plot(Time,180*X[3,:]/np.pi,'g--')
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
                            Color='g',
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
                            Color='g',
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

def animate_trajectory(Time,X,U,**kwargs):
    SaveAsGif = kwargs.get("SaveAsGif",False)
    assert type(SaveAsGif)==bool, "SaveAsGif must be either True or False (Default)."

    FileName = kwargs.get("FileName","ddp_simple_pendulum")
    assert type(FileName)==str,"FileName must be a str."

        # Angles must be in degrees for animation

    X2d = X[1,:]*(180/np.pi)
    X4d = X[3,:]*(180/np.pi)


    fig = plt.figure(figsize=(12,10))
    ax0 = plt.subplot2grid((3,4),(0,0),colspan=2) # animation
    ax1 = plt.subplot2grid((3,4),(0,2),colspan=2) # input
    ax2 = plt.subplot2grid((3,4),(1,0),colspan=2) # pendulum angle
    ax3 = plt.subplot2grid((3,4),(1,2),colspan=2) # cart position
    ax4 = plt.subplot2grid((3,4),(2,0),colspan=2) # pendulum angular velocity
    ax5 = plt.subplot2grid((3,4),(2,2),colspan=2) # cart velocty

    plt.suptitle("Cart-Pendulum Example",Fontsize=28,y=0.95)

    Pendulum_Width = 0.01*L
    Pendulum_Length = 2*L
    Cart_Width = 4*L
    Cart_Height = 2*L
    Wheel_Radius = 0.125*Cart_Width
    # Model Drawing
    marker_interdistance = 25
    lowest_marker = marker_interdistance* \
            (int(np.floor(X[0].min())/marker_interdistance)-1)
    highest_marker = marker_interdistance* \
            (int(np.ceil(X[0].max())/marker_interdistance)+2)
    markers = np.arange(
            lowest_marker,
            highest_marker,
            marker_interdistance
            )
    smallmarkers = np.arange(
            lowest_marker,
            highest_marker,
            marker_interdistance/2
            )
    marker_str = []
    for marker in markers:
        if marker%100==0:
            marker_str.append(str(int(marker)))
        else:
            marker_str.append("")

    Markers = ax0.scatter(
                markers-X[0,0],
                -0.90*(Cart_Height/2 + Wheel_Radius*2)*np.ones(len(markers)),
                marker="|",
                c="k",
                s=2000)

    SmallMarkers = ax0.scatter(
                smallmarkers-X[0,0],
                -0.90*(Cart_Height/2
                    + Wheel_Radius*2)*np.ones(len(smallmarkers)),
                marker="|",
                c="k",
                s=1000)

    Cart = plt.Rectangle(
                (X[0,0]-Cart_Width/2,-Cart_Height/2),
                Cart_Width,
                Cart_Height,
                Color='#4682b4')
    ax0.add_patch(Cart)

    FrontWheel = plt.Circle(
                (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                Wheel_Radius,
                Color='k')
    ax0.add_patch(FrontWheel)
    FrontWheel_Rivet = plt.Circle(
                (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                0.2*Wheel_Radius,
                Color='0.70')
    ax0.add_patch(FrontWheel_Rivet)

    BackWheel = plt.Circle(
                (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                Wheel_Radius,
                Color='k')
    ax0.add_patch(BackWheel)
    BackWheel_Rivet = plt.Circle(
                (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                0.2*Wheel_Radius,
                Color='0.70')
    ax0.add_patch(BackWheel_Rivet)

    Pendulum, = ax0.plot(
                    [
                        X[0,0],
                        X[0,0] + Pendulum_Length*np.sin(X[1,0])
                    ],
                    [
                        Cart_Height/2 + Pendulum_Width/2,
                        Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(X[1,0])
                    ],
                    Color='0.50',
                    lw = 5,
                    solid_capstyle='round'
                    )


    Pendulum_Attachment = plt.Circle((X[0,0],Cart_Height/2),100*Pendulum_Width/2,Color='#4682b4')
    ax0.add_patch(Pendulum_Attachment)

    Pendulum_Rivet, = ax0.plot(
        [X[0,0]],
        [Cart_Height/2 + Pendulum_Width/2],
        c='k',
        marker='o',
        lw=1
        )

    MinimumX = X[0,0]-13
    MaximumX = X[0,0]+13
    # if max(abs(X[1,:]-X[1,0]))<1e-7:
    #     MinimumX = X[1,0]-10
    #     MaximumX = X[1,0]+10
    # elif max(abs(X[1,:]-X[1,0]))<2:
    #     MinimumX = min(X[1,:])-1.25*Cart_Width/2-5
    #     MaximumX = max(X[1,:])+1.25*Cart_Width/2+5
    # else:
    #     MinimumX = min(X[1,:])-1.25*Cart_Width/2
    #     MaximumX = max(X[1,:])+1.25*Cart_Width/2

    Ground = plt.Rectangle(
                (MinimumX,-1.50*(Cart_Height/2 + Wheel_Radius*2)),
                MaximumX-MinimumX,
                0.50*(Cart_Height/2 + Wheel_Radius*2),
                Color='0.70')
    ax0.add_patch(Ground)

    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax0.set_frame_on(True)

    ax0.set_xlim([MinimumX,MaximumX])
    ax0.set_ylim(
        [
            -1.50*(Cart_Height/2 + Wheel_Radius*2),
            1.50*(Cart_Height/2+Pendulum_Width/2+Pendulum_Length+Pendulum_Width/2)
        ]
        )
    ax0.set_aspect('equal')

    #Input

    Input, = ax1.plot([0],[U[0]],color = 'b')
    ax1.set_xlim(0,Time[-1])
    ax1.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax1.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(U-U[0]))<1e-7:
        ax1.set_ylim([U[0]-2,U[0]+2])
    else:
        RangeU = max(U)-min(U)
        ax1.set_ylim([min(U)-0.1*RangeU,max(U)+0.1*RangeU])

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_title("Input Force (N)",fontsize=16,fontweight = 4,color = 'b',y = 0.95)

    #Pendulum Angle

    Angle, = ax2.plot([0],[X2d[0]],color = 'r')
    ax2.set_xlim(0,Time[-1])
    ax2.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax2.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X2d-X2d[0]))<1e-7:
        ax2.set_ylim([X2d[0]-2,X2d[0]+2])
    else:
        RangeX2d= max(X2d)-min(X2d)
        ax2.set_ylim([min(X2d)-0.1*RangeX2d,max(X2d)+0.1*RangeX2d])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title("Pendulum Angle (deg)",fontsize=16,fontweight = 4,color = 'r',y = 0.95)

    #Cart Position

    Position, = ax3.plot([0],[X[0,0]],color = 'g')
    ax3.set_xlim(0,Time[-1])
    ax3.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax3.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X[0,:]-X[0,0]))<1e-7:
        ax3.set_ylim([X[0,0]-2,X[0,0]+2])
    else:
        RangeX1 = max(X[0,:])-min(X[0,:])
        ax3.set_ylim([min(X[0,:])-0.1*RangeX1,max(X[0,:])+0.1*RangeX1])

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.set_title("Cart Position (m)",fontsize=16,fontweight = 4,color = 'g',y = 0.95)

    # Angular Velocity

    AngularVelocity, = ax4.plot([0],[X4d[0]],color='r',linestyle='--')
    ax4.set_xlim(0,Time[-1])
    ax4.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax4.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X4d-X4d[0]))<1e-7:
        ax4.set_ylim([X4d[0]-2,X4d[0]+2])
    else:
        RangeX4d= max(X4d)-min(X4d)
        ax4.set_ylim([min(X4d)-0.1*RangeX4d,max(X4d)+0.1*RangeX4d])
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_title("Pendulum Angular Velocity (deg/s)",fontsize=16,fontweight = 4,color = 'r',y = 0.95)

    # Cart Velocity

    Velocity, = ax5.plot([0],[X[2,0]],color='g',linestyle="--")
    ax5.set_xlim(0,Time[-1])
    ax5.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax5.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X[2,:]-X[2,0]))<1e-7:
        ax5.set_ylim([X[2,0]-2,X[2,0]+2])
    else:
        RangeX3= max(X[2,:])-min(X[2,:])
        ax5.set_ylim([min(X[2,:])-0.1*RangeX3,max(X[2,:])+0.1*RangeX3])
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.set_title("Cart Velocity (m/s)",fontsize=16,fontweight = 4,color = 'g',y = 0.95)

    def animate(i):
        # Cart.xy = (X[1,i]-Cart_Width/2,-Cart_Height/2)
        #
        # FrontWheel.center = (X[1,i]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        # FrontWheel_Rivet.center = (X[1,i]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        #
        # BackWheel.center = (X[1,i]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        # BackWheel_Rivet.center = (X[1,i]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        #
        # Pendulum.set_xdata([X[1,i],X[1,i] + Pendulum_Length*np.sin(X[0,i])])
        # Pendulum.set_ydata([Cart_Height/2 + Pendulum_Width/2,
        #                     Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(X[0,i])])
        #
        # Pendulum_Attachment.center = (X[1,i],Cart_Height/2)
        #
        # Pendulum_Rivet.set_xdata([X[1,i]])

        offset = np.concatenate([
                (markers-X[0,i])[:,np.newaxis],
                (-0.90*(Cart_Height/2 + Wheel_Radius*2)
                    * np.ones(len(markers)))[:,np.newaxis]
                ],
                axis=1)
        Markers.set_offsets(offset)

        smalloffset = np.concatenate([
                (smallmarkers-X[0,i])[:,np.newaxis],
                (-0.90*(Cart_Height/2 + Wheel_Radius*2)
                    * np.ones(len(smallmarkers)))[:,np.newaxis]
                ],
                axis=1)
        SmallMarkers.set_offsets(smalloffset)

        Cart.xy = (X[0,0]-Cart_Width/2,-Cart_Height/2)

        FrontWheel.center = (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        FrontWheel_Rivet.center = (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))

        BackWheel.center = (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        BackWheel_Rivet.center = (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))

        Pendulum.set_xdata([X[0,0],X[0,0] + Pendulum_Length*np.sin(X[1,i])])
        Pendulum.set_ydata([Cart_Height/2 + Pendulum_Width/2,
                            Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(X[1,i])])

        Pendulum_Attachment.center = (X[0,0],Cart_Height/2)

        Pendulum_Rivet.set_xdata([X[0,0]])

        Input.set_xdata(Time[:i])
        Input.set_ydata(U[:i])

        Position.set_xdata(Time[:i])
        Position.set_ydata(X[0,:i])

        Angle.set_xdata(Time[:i])
        Angle.set_ydata(X2d[:i])

        Velocity.set_xdata(Time[:i])
        Velocity.set_ydata(X[2,:i])

        AngularVelocity.set_xdata(Time[:i])
        AngularVelocity.set_ydata(X4d[:i])

        return SmallMarkers,Markers,Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Input,Position,Angle,Velocity,AngularVelocity,

    # Init only required for blitting to give a clean slate.
    def init():
        Markers = plt.scatter(
                    markers-X[0,0],
                    -0.90*(Cart_Height/2 + Wheel_Radius*2)*np.ones(len(markers)),
                    marker="^",
                    c="k",
                    s=2000)

        SmallMarkers = plt.scatter(
                    smallmarkers-X[0,0],
                    -0.90*(Cart_Height/2 + Wheel_Radius*2)*np.ones(len(smallmarkers)),
                    marker="^",
                    c="k",
                    s=1000)

        Cart = plt.Rectangle(
                    (X[0,0]-Cart_Width/2,-Cart_Height/2),
                    Cart_Width,
                    Cart_Height,
                    Color='#4682b4')
        ax0.add_patch(Cart)

        FrontWheel = plt.Circle(
                    (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    Wheel_Radius,
                    Color='k')
        ax0.add_patch(FrontWheel)
        FrontWheel_Rivet = plt.Circle(
                    (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    0.2*Wheel_Radius,
                    Color='0.70')
        ax0.add_patch(FrontWheel_Rivet)

        BackWheel = plt.Circle(
                    (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    Wheel_Radius,
                    Color='k')
        ax0.add_patch(BackWheel)
        BackWheel_Rivet = plt.Circle(
                    (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    0.2*Wheel_Radius,
                    Color='0.70')
        ax0.add_patch(BackWheel_Rivet)

        Pendulum, = ax0.plot(
                        [
                            X[0,0],
                            X[0,0] + Pendulum_Length*np.sin(X[1,0])
                        ],
                        [
                            Cart_Height/2 + Pendulum_Width/2,
                            Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(X[1,0])
                        ],
                        Color='0.50',
                        lw = 10,
                        solid_capstyle='round'
                        )

        Pendulum_Attachment = plt.Circle((X[0,0],Cart_Height/2),100*Pendulum_Width/2,Color='#4682b4')
        ax0.add_patch(Pendulum_Attachment)

        Pendulum_Rivet, = ax0.plot(
            [X[0,0]],
            [Cart_Height/2 + Pendulum_Width/2],
            c='k',
            marker='o',
            lw=1
            )

        Ground = plt.Rectangle(
                    (MinimumX,-1.50*(Cart_Height/2 + Wheel_Radius*2)),
                    MaximumX-MinimumX,
                    0.50*(Cart_Height/2 + Wheel_Radius*2),
                    Color='0.70')
        ax0.add_patch(Ground)

        #Input

        Input, = ax1.plot([0],[U[0]],color = 'b')

        #Pendulum Angle

        Angle, = ax2.plot([0],[X2d[0]],color = 'r')

        #Cart Position

        Position, = ax3.plot([0],[X[0,0]],color = 'g')

        # Angular Velocity

        AngularVelocity, = ax4.plot([0],[X4d[0]],color = 'r')

        # Cart Velocity

        Velocity, = ax5.plot([0],[X[2,0]],color = 'g--')

        Markers.set_visible(False)
        SmallMarkers.set_visible(False)
        Cart.set_visible(False)
        FrontWheel.set_visible(False)
        FrontWheel_Rivet.set_visible(False)
        BackWheel.set_visible(False)
        BackWheel_Rivet.set_visible(False)
        Pendulum.set_visible(False)
        Pendulum_Attachment.set_visible(False)
        Pendulum_Rivet.set_visible(False)
        Ground.set_visible(True)
        Input.set_visible(False)
        Position.set_visible(False)
        Angle.set_visible(False)
        Velocity.set_visible(False)
        AngularVelocity.set_visible(False)

        return SmallMarkers,Markers,Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Ground,Input,Position,Angle,Velocity,

    dt = Time[1]-Time[0]
    if dt <= 0.0001:
        framestep=2000
    elif dt <= 0.001:
        framestep=200
    elif dt <= 0.01:
        framestep=10
    else:
        framestep=1
    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(Time)-1,framestep),init_func=init, blit=False)
    if SaveAsGif==True:
        ani.save("visualizations_cart_pendulum/"+FileName+'.gif', writer='imagemagick', fps=10)
    plt.show()

def return_f32(X,U):
    return(
        (
            (
                m2*L*np.cos(X[1])*(X[3]**2)
                - m2*g*np.cos(2*X[1])
                - b2*np.sin(X[1])*X[3]/L
            )
            *
            (
                m1 + m2*(np.sin(X[1])**2)
            )
            -
            (
                m2*L*np.sin(X[1])*(X[3]**2)
                - m2*g*np.sin(2*X[1])/2
                - b1*X[2]
                + U
                + b2*np.cos(X[1])*X[3]/L
            )
            *
            (
                m2*np.sin(2*X[1])
            )
        )
        /
        (
            (
                m1 + m2*(np.sin(X[1])**2)
            )**2
        )
    )
def return_f33(X,U):
    return(
        (
            -b1
        )
        /
        (
            m1 + m2*(np.sin(X[1])**2)
        )
    )
def return_f34(X,U):
    return(
        (
            2*m2*L*np.sin(X[1])*X[3]
            + b2*np.cos(X[1])/L
        )
        /
        (
            m1 + m2*(np.sin(X[1])**2)
        )
    )

def return_f42(X,U):
    return(
        (
            (
                -m2*np.cos(2*X[1])*(X[3]**2)
                + (m1+m2)*g*np.cos(X[1])/L
                - b1*np.sin(X[1])*X[2]/L
                + np.sin(X[1])*U/L
            )
            *
            (
                m1 + m2*(np.sin(X[1])**2)
            )
            -
            (
                -m2*np.sin(2*X[1])*(X[3]**2)/2
                + (m1+m2)*g*np.sin(X[1])/L
                + b1*np.cos(X[1])*X[2]/L
                - (m1+m2)*b2*X[3]/(m2*(L**2))
                - np.cos(X[1])*U/L
            )
            *
            (
                m2*np.sin(2*X[1])
            )
        )
        /
        (
            (
                m1 + m2*(np.sin(X[1])**2)
            )**2
        )
    )
def return_f43(X,U):
    return(
        (
            b1*np.cos(X[1])/L
        )
        /
        (
            m1 + m2*(np.sin(X[1])**2)
        )
    )
def return_f44(X,U):
    return(
        (
            -m2*np.sin(2*X[1])*X[3]
            - (m1+m2)*b2/(m2*(L**2))
        )
        /
        (
            m1 + m2*(np.sin(X[1])**2)
        )
    )

def return_Phi(X,U,dt):
    """
    Takes in the state vector, X, of shape (4,) and a number U, and outputs a matrix of shape (4,4)
    """
    assert np.shape(X)==(4,) and str(type(X))=="<class 'numpy.ndarray'>", \
        "X must be an numpy array of shape (4,)"
    assert str(type(U)) in ["<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.float'>",
            "<class 'numpy.float64'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>"],\
        "U must be a number. Not " + str(type(U)) + "."
    result = (np.eye(4)
        + np.matrix(
            [
            [0, 0, dt, 0],
            [0, 0, 0, dt],
            [0, return_f32(X,U)*dt, return_f33(X,U)*dt, return_f34(X,U)*dt],
            [0, return_f42(X,U)*dt, return_f43(X,U)*dt, return_f44(X,U)*dt]
            ]
        )
    )

    assert np.shape(result)==(4,4) \
            and str(type(result))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
        "result must be a (4,4) numpy matrix. Not " + str(type(result)) + " of shape " + str(np.shape(result)) + "."

    return(result)
def return_B(X,U,dt):
    """
    Takes in the state vector, X, of shape (4,) and a number U, and outputs a matrix of shape (4,1)
    """
    assert np.shape(X)==(4,) and str(type(X))=="<class 'numpy.ndarray'>", \
        "X must be an numpy array of shape (4,)"
    assert str(type(U)) in ["<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.float'>",
            "<class 'numpy.float64'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>"],\
        "U must be a number. Not " + str(type(U)) + "."
    result = (
        np.matrix(
            [
            [0],
            [0],
            [(dt)/(m1 + m2*(np.sin(X[1])**2))],
            [(-np.cos(X[1])*dt/L)/(m1 + m2*(np.sin(X[1])**2))]
            ]
        )
    )

    assert np.shape(result)==(4,1) \
            and str(type(result))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
        "result must be a (4,1) numpy matrix. Not " + str(type(result)) + " of shape " + str(np.shape(result)) + "."

    return(result)
def return_linearized_dynamics_matrices(X,U,dt):
    """
    Takes in the input U and the the corresponding output X, as well as dt and returns two lists that contain the linearized dynamic matrices for each timestep for range(len(Time)-1).

    Note that if np.shape(X)[1] = N and len(U) = M, then N = M + 1 (i.e., there is one more timestep for output than input since the initial conditions are assigned to the first state space timestep). Therefore, we only concern ourselves with the linearized dynamics of the (N-1) steps where U drives X to the next timestep (i.e., X will only go up to the N-1 step or index X[:,:-1].)

    Phi is a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.

    B is a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.

    ### NEEDS TO BE TESTED ###

    np.shape(X)[1] == len(U)+1

    len(Phi) == len(U)
    type(Phi) == list
    len(B) == len(U)
    type(B) == list

    ##########################
    """
    Phi = list(
            map(
                lambda X,U: return_Phi(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    B = list(
            map(
                lambda X,U: return_B(X,U,dt),
                X[:,:-1].T,
                U
            )
        )
    return(Phi,B)

def return_l_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[(d1/2)*U**2]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[c2*(1/2)*(X[1]-TargetAngle)**2]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0]])
    # if "Minimize time away from target angle" in RunningCost:
    #     """
    #     In order to approximate (k2/2)*(X[1]-TargetAngle)**2 that repeats every 2*np.pi we must have a time shifted fourier series expansion. c2*(np.pi**2)/6 + sum([((-1)**n) * (2*c2/(n**2)) * np.cos(n*(X[1]-TargetAngle)) for n in range(1,N)])
    #     """
    #     result2 = lambda X,U,dt: np.matrix([
    #             [
    #                 c2*(np.pi**2)/6
    #                 + sum([
    #                         ((-1)**n)
    #                         *(2*c2/(n**2))
    #                         *np.cos(n*(X[1]-TargetAngle))
    #                     for n in range(1,100)
    #                 ])
    #             ]
    #         ])
    # else:
    #     result2 = lambda X,U,dt: np.matrix([[0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt:\
                    np.matrix([[c4*(1/2)*(X[3]-TargetAngularVelocity)**2]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0]])

    if "Minimize time away from initial position" in RunningCost:
        result4 = lambda X,U,dt:\
                    np.matrix([[c1*(1/2)*(X[0])**2]])
    else:
        result4 = lambda X,U,dt: np.matrix([[0]])

    result = lambda X,U,dt: (
                            result1(X,U,dt)
                            + result2(X,U,dt)
                            + result3(X,U,dt)
                            + result4(X,U,dt)
                            ) * dt
    return(result)
def return_lx_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."


    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[0],[c2*(X[1]-TargetAngle)],[0],[0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])
    # if "Minimize time away from target angle" in RunningCost:
    #     """
    #     In order to approximate (k2/2)*(X[1]-TargetAngle)**2 that repeats every 2*np.pi we must have a time shifted fourier series expansion. c2*(np.pi**2)/6 + sum([((-1)**n) * (2*c2/(n**2)) * np.cos(n*(X[1]-TargetAngle)) for n in range(1,N)])
    #     """
    #     result2 = lambda X,U,dt: np.matrix([
    #             [0],
    #             [
    #                 sum([
    #                         ((-1)**(n+1))
    #                         *(2*c2/n)
    #                         *np.sin(n*(X[1]-TargetAngle))
    #                     for n in range(1,100)
    #                 ])
    #             ],
    #             [0],
    #             [0]
    #         ])
    # else:
    #     result2 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0],[0],[0],[c4*(X[3]-TargetAngularVelocity)]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])

    if "Minimize time away from initial position" in RunningCost:
        result4 = lambda X,U,dt: np.matrix([[c1*(X[0])],[0],[0],[0]])
    else:
        result4 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])

    result = lambda X,U,dt: (
                            result1(X,U,dt)
                            + result2(X,U,dt)
                            + result3(X,U,dt)
                            + result4(X,U,dt)
                            ) * dt
    return(result)
def return_lu_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[d1*U]])
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

    if "Minimize time away from initial position" in RunningCost:
        result4 = lambda X,U,dt: np.matrix([[0]])
    else:
        result4 = lambda X,U,dt: np.matrix([[0]])

    result = lambda X,U,dt: (
                            result1(X,U,dt)
                            + result2(X,U,dt)
                            + result3(X,U,dt)
                            + result4(X,U,dt)
                            ) * dt
    return(result)
def return_lxu_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])
    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])
    if "Minimize time away from initial position" in RunningCost:
        result4 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])
    else:
        result4 = lambda X,U,dt: np.matrix([[0],[0],[0],[0]])

    result = lambda X,U,dt: (
                            result1(X,U,dt)
                            + result2(X,U,dt)
                            + result3(X,U,dt)
                            + result4(X,U,dt)
                            ) * dt
    return(result)
def return_lux_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[0,0,0,0]])
    else:
        result1 = lambda X,U,dt: np.matrix([[0,0,0,0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix([[0,0,0,0]])
    else:
        result2 = lambda X,U,dt: np.matrix([[0,0,0,0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix([[0,0,0,0]])
    else:
        result3 = lambda X,U,dt: np.matrix([[0,0,0,0]])

    if "Minimize time away from initial position" in RunningCost:
        result4 = lambda X,U,dt: np.matrix([[0,0,0,0]])
    else:
        result4 = lambda X,U,dt: np.matrix([[0,0,0,0]])

    result = lambda X,U,dt: (
                            result1(X,U,dt)
                            + result2(X,U,dt)
                            + result3(X,U,dt)
                            + result4(X,U,dt)
                            ) * dt
    return(result)
def return_luu_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix([[d1]])
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
    if "Minimize time away from initial position" in RunningCost:
        result4 = lambda X,U,dt: np.matrix([[0]])
    else:
        result4 = lambda X,U,dt: np.matrix([[0]])

    result = lambda X,U,dt: (
                            result1(X,U,dt)
                            + result2(X,U,dt)
                            + result3(X,U,dt)
                            + result4(X,U,dt)
                            ) * dt
    return(result)
def return_lxx_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.matrix(
                                    [[0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0]])
    else:
        result1 = lambda X,U,dt: np.matrix(
                                    [[0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0]])

    if "Minimize time away from target angle" in RunningCost:
        result2 = lambda X,U,dt: np.matrix(
                                    [[0,0,0,0],
                                    [0,c2,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0]])
    else:
        result2 = lambda X,U,dt: np.matrix(
                                    [[0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0]])
    # if "Minimize time away from target angle" in RunningCost:
    #     """
    #     In order to approximate (k2/2)*(X[1]-TargetAngle)**2 that repeats every 2*np.pi we must have a time shifted fourier series expansion. c2*(np.pi**2)/6 + sum([((-1)**n) * (2*c2/(n**2)) * np.cos(n*(X[1]-TargetAngle)) for n in range(1,N)])
    #
    #     NOTE: This does something funny when you use more than 100 frequencies. The limit of this function approaches 2*c2 not c2. If this creates undesireable features, then it might be best to change this value to c2 exclusively.
    #     """
    #     result2 = lambda X,U,dt: np.matrix(
    #                                 [[0,0,0,0],
    #                                 [
    #                                     0,
    #                                     sum([
    #                                             ((-1)**(n+1))
    #                                             *(2*c2)
    #                                             *np.cos(n*(X[1]-TargetAngle))
    #                                         for n in range(1,100)
    #                                         ]),
    #                                     0,
    #                                     0
    #                                 ],
    #                                 [0,0,0,0],
    #                                 [0,0,0,0]])
    # else:
    #     result2 = lambda X,U,dt: np.matrix(
    #                                 [[0,0,0,0],
    #                                 [0,0,0,0],
    #                                 [0,0,0,0],
    #                                 [0,0,0,0]])

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt: np.matrix(
                                    [[0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,c4]])
    else:
        result3 = lambda X,U,dt: np.matrix(
                                    [[0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0]])

    if "Minimize time away from initial position" in RunningCost:
        result4 = lambda X,U,dt: np.matrix(
                                    [[c1,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0]])
    else:
        result4 = lambda X,U,dt: np.matrix(
                                    [[0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0],
                                    [0,0,0,0]])

    result = lambda X,U,dt: (
                            result1(X,U,dt)
                            + result2(X,U,dt)
                            + result3(X,U,dt)
                            + result4(X,U,dt)
                            ) * dt
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
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."

    # returns a list of length len(Time)-1, each element with shape (1,1), where n is the number of states.
    l_func = return_l_func(RunningCost=RunningCost)
    l = list(
            map(
                lambda X,U: l_func(X,U,dt),
                X[:,1:].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,1), where n is the number of states.
    lx_func = return_lx_func(RunningCost=RunningCost)
    lx = list(
            map(
                lambda X,U: lx_func(X,U,dt),
                X[:,1:].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,1), where n is the number of states.
    lu_func = return_lu_func(RunningCost=RunningCost)
    lu = list(
            map(
                lambda X,U: lu_func(X,U,dt),
                X[:,1:].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,n), where m is the number of inputs and n is the number of states.
    lux_func = return_lux_func(RunningCost=RunningCost)
    lux = list(
            map(
                lambda X,U: lux_func(X,U,dt),
                X[:,1:].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
    lxu_func = return_lxu_func(RunningCost=RunningCost)
    lxu = list(
            map(
                lambda X,U: lxu_func(X,U,dt),
                X[:,1:].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,m), where m is the number of inputs.
    luu_func = return_luu_func(RunningCost=RunningCost)
    luu = list(
            map(
                lambda X,U: luu_func(X,U,dt),
                X[:,1:].T,
                U
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
    lxx_func = return_lxx_func(RunningCost=RunningCost)
    lxx = list(
            map(
                lambda X,U: lxx_func(X,U,dt),
                X[:,1:].T,
                U
            )
        )

    return(l,lx,lu,lux,lxu,luu,lxx)

def return_running_cost_func(RunningCost='Minimize Input Energy'):
    """
    Should use the current timestep, not the t + dt (or prime notation).

    RunningCost should be either 'Minimize Input Energy' (Default), 'Minimize time away from target angle', or 'Minimize time away from target angular velocity'. To be set upstream by linearized cost function.
    """
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."

    if "Minimize Input Energy" in RunningCost:
        result1 = lambda X,U,dt: np.trapz((d1/2)*U**2,dx=dt)
    else:
        result1 = lambda X,U,dt: 0

    if "Minimize time away from target angle" in RunningCost:
        """
        In order to approximate (k2/2)*(X[1]-TargetAngle)**2 that repeats every 2*np.pi we must have a time shifted fourier series expansion. c2*(np.pi**2)/6 + sum([((-1)**n) * (2*c2/(n**2)) * np.cos(n*(X[1]-TargetAngle)) for n in range(1,N)])

        Note: we checked to make sure that the linearity holds for this trapz/sum combination and it does.
        """
        result2 = lambda X,U,dt: np.trapz(c2*(1/2)*(X[1,1:]-TargetAngle)**2,dx=dt)
        # result2 = lambda X,U,dt: np.trapz(
        #     c2*(np.pi**2)/6
        #     + sum([
        #             ((-1)**n)
        #             *(2*c2/(n**2))
        #             *np.cos(n*(X[1,1:]-TargetAngle))
        #         for n in range(1,100)
        #     ]),
        #     dx=dt
        # )
    else:
        result2 = lambda X,U,dt: 0

    if "Minimize time away from target angular velocity" in RunningCost:
        result3 = lambda X,U,dt:\
                    np.trapz(c4*(1/2)*(X[3,1:]-TargetAngularVelocity)**2,dx=dt)
    else:
        result3 = lambda X,U,dt: 0

    if "Minimize time away from initial position" in RunningCost:
        result4 = lambda X,U,dt:\
                    np.trapz(c1*(1/2)*(X[0,1:])**2,dx=dt)
    else:
        result4 = lambda X,U,dt: 0

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt) \
                            + result3(X,U,dt) \
                            + result4(X,U,dt)
    return(result)

def return_terminal_cost_func(TerminalCost='Minimize final angle',
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

    if "Minimize final angle from target angle" in TerminalCost:
        """
        In order to approximate (k2/2)*(X[1]-TargetAngle)**2 that repeats every 2*np.pi we must have a time shifted fourier series expansion. c2*(np.pi**2)/6 + sum([((-1)**n) * (2*c2/(n**2)) * np.cos(n*(X[1]-TargetAngle)) for n in range(1,N)])
        """
        result1 = lambda X,U,dt: k2*(1/2)*(X[1,-1]-TargetAngle)**2
        result1_grad = lambda X,U,dt:\
            np.matrix([[0],[k2*(X[1,-1]-TargetAngle)],[0],[0]])
        result1_hess = lambda X,U,dt: np.matrix(
                                [[0,0,0,0],
                                [0,k2,0,0],
                                [0,0,0,0],
                                [0,0,0,0]])
        # result1 = lambda X,U,dt: (
        #             k2*(np.pi**2)/6
        #             + sum([
        #                     ((-1)**n)
        #                     *(2*k2/(n**2))
        #                     *np.cos(n*(X[1,-1]-TargetAngle))
        #                 for n in range(1,100)
        #             ])
        #         )
        # result1_grad = lambda X,U,dt:\
        #     np.matrix([
        #             [0],
        #             [
        #                 sum([
        #                         ((-1)**(n+1))
        #                         *(2*k2/n)
        #                         *np.sin(n*(X[1,-1]-TargetAngle))
        #                     for n in range(1,100)
        #                 ])
        #             ],
        #             [0],
        #             [0]
        #         ])
        # result1_hess = lambda X,U,dt: np.matrix(
        #                             [[0,0,0,0],
        #                             [
        #                                 0,
        #                                 sum([
        #                                         ((-1)**(n+1))
        #                                         *(2*c2)
        #                                         *np.cos(n*(X[1,-1]-TargetAngle))
        #                                     for n in range(1,100)
        #                                     ]),
        #                                 0,
        #                                 0
        #                             ],
        #                             [0,0,0,0],
        #                             [0,0,0,0]])
    else:
        result1 = lambda X,U,dt: 0
        result1_grad = lambda X,U,dt:\
            np.matrix([[0],[0],[0],[0]])
        result1_hess = lambda X,U,dt: np.matrix(
                                [[0,0,0,0],
                                [0,0,0,0],
                                [0,0,0,0],
                                [0,0,0,0]])

    if "Minimize final angular velocity from target angular velocity" in TerminalCost:
        result2 = lambda X,U,dt: k4*(1/2)*(X[3,-1]-TargetAngularVelocity)**2
        result2_grad = lambda X,U,dt:\
            np.matrix([[0],[0],[0],[k4*(X[3,-1]-TargetAngularVelocity)]])
        result2_hess = lambda X,U,dt: np.matrix(
                        [[0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,k4]])
    else:
        result2 = lambda X,U,dt: 0
        result2_grad = lambda X,U,dt:\
            np.matrix([[0],[0],[0],[0]])
        result2_hess = lambda X,U,dt: np.matrix(
                                [[0,0,0,0],
                                [0,0,0,0],
                                [0,0,0,0],
                                [0,0,0,0]])

    result = lambda X,U,dt: result1(X,U,dt) \
                            + result2(X,U,dt)
    if ReturnGradientAndHessian:
        result_grad = lambda X,U,dt: result1_grad(X,U,dt) \
                                        + result2_grad(X,U,dt)
        result_hess = lambda X,U,dt: result1_hess(X,U,dt) \
                                        + result2_hess(X,U,dt)
        return(result,result_grad,result_hess)
    else:
        return(result)

def return_empty_lists_for_quadratic_expansion_of_Q(length):
    Qu = [None]*length
    Qx = [None]*length
    Qux = [None]*length
    Qxu = [None]*length
    Quu = [None]*length
    Qxx = [None]*length
    return(Qu,Qx,Qux,Qxu,Quu,Qxx)

def cart_pendulum_ddp(**kwargs):

    RunningCost = kwargs.get("RunningCost","Minimize Input Energy")
    if type(RunningCost)==str:
        assert RunningCost in ['Minimize Input Energy',
                    'Minimize time away from target angle',
                    'Minimize time away from target angular velocity',
                    'Minimize time away from initial position'],\
            "RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'."
    else:
        assert type(RunningCost)==list, "RunningCost must be a list of cost types."
        for el in RunningCost:
            assert type(el)==str, "Each element of RunningCost must be a string. Not " + str(type(el)) + "."
            assert el in ['Minimize Input Energy',
                        'Minimize time away from target angle',
                        'Minimize time away from target angular velocity',
                        'Minimize time away from initial position'],\
                "Each element of RunningCost must be either 'Minimize Input Energy','Minimize time away from target angle', 'Minimize time away from target angular velocity', or 'Minimize time away from initial position'. '" + el + "' not accepted."
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
    terminal_cost_func,terminal_cost_grad_func,terminal_cost_hess_func = \
        return_terminal_cost_func(
            TerminalCost=TerminalCost,
            ReturnGradientAndHessian=True
            )

    ICs = kwargs.get("ICs",[0,0,0,0]) # in degrees
    assert type(ICs)==list and len(ICs)==4, "ICs must be a list of length 4."
    LocationStrings = ["1st", "2nd", "3rd", "4th"]
    for i in range(4):
        assert str(type(ICs[i])) in [
                "<class 'numpy.float'>",
                "<class 'int'>",
                "<class 'float'>",
                "<class 'numpy.int32'>",
                "<class 'numpy.int64'>",
                "<class 'numpy.float64'>"],\
            "ICs must be numbers. Check the " + LocationStrings[i] + " element of IC"

    dt = kwargs.get("dt",0.01)
    assert str(type(dt)) in [
            "<class 'numpy.float'>",
            "<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>",
            "<class 'numpy.float64'>"],\
        "dt must be a number."

    N_seconds = kwargs.get("N_seconds",10)
    assert str(type(N_seconds)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
        "N_seconds must be a number."

    N_iterations = kwargs.get("N_iterations",10)
    assert str(type(N_iterations)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
        "N_iterations must be a number."

    Animate = kwargs.get("Animate",True)
    assert type(Animate)==bool, "Animate must be either True (Default) or False."

    PlotCost = kwargs.get("PlotCost",True)
    assert type(PlotCost)==bool, "PlotCost must be either True (Default) or False."

    thresh = kwargs.get("thresh",1e-2)
    assert str(type(thresh)) in ["<class 'numpy.float'>","<class 'int'>","<class 'float'>","<class 'numpy.int32'>","<class 'numpy.int64'>","<class 'numpy.float64'>"],\
        "thresh must be a number."


    TotalX = []
    TotalU = []

    Time = np.arange(0,N_seconds + dt,dt)

    U = kwargs.get("U",0*np.ones(len(Time)-1)) # initial input
    assert len(U)==len(Time)-1, "U must be an array with length len(Time)-1."

    TrialCosts = []
    IterationNumber = 1
    ThresholdNotMet=True
    while IterationNumber <= N_iterations and ThresholdNotMet:
        print(
            "Iteration Number : "
            + str(IterationNumber)
            + "/"
            + str(N_iterations)
        ) # Does not effect runtime

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
        if IterationNumber>3:
            if abs(np.average(TrialCosts[-3:])-TrialCosts[-1]) < thresh:
                ThresholdNotMet = False
                print("Threshold met after " + str(IterationNumber) + " Trials")

        Phi,B = return_linearized_dynamics_matrices(X,U,dt)
        l,lx,lu,lux,lxu,luu,lxx = \
                return_quadratic_cost_function_expansion_variables(X,U,dt,RunningCost=RunningCost)

        # Backward Pass
        V = [np.matrix([[terminal_cost_func(X,U,dt)]])]
        Vx = [terminal_cost_grad_func(X,U,dt)]
        Vxx = [terminal_cost_hess_func(X,U,dt)]

        Qu,Qx,Qux,Qxu,Quu,Qxx = \
                return_empty_lists_for_quadratic_expansion_of_Q(len(Time)-1)

        for i in range(len(Time)-1):
            Qx[-(i+1)] = lx[-(i+1)] + Phi[-(i+1)].T*Vx[-1]
            Qu[-(i+1)] = lu[-(i+1)] + B[-(i+1)].T*Vx[-1]
            Qux[-(i+1)] = lux[-(i+1)] + B[-(i+1)].T*Vxx[-1]*Phi[-(i+1)]
            Qxu[-(i+1)] = lxu[-(i+1)] + Phi[-(i+1)].T*Vxx[-1]*B[-(i+1)]
            Quu[-(i+1)] = luu[-(i+1)] + B[-(i+1)].T*Vxx[-1]*B[-(i+1)]
            Qxx[-(i+1)] = lxx[-(i+1)] + Phi[-(i+1)].T*Vxx[-1]*Phi[-(i+1)]


            # It appears that the choice of this does not matter... Ran it for 5 seconds, Target = pi, and the final cost was always 621.50
            V.append(
                    l[-(i+1)]
                    + V[-1]
                    - (1/2)*Qu[-(i+1)].T*(Quu[-(i+1)]**(-1))*Qu[-(i+1)]
                )
            # V.append(
            #          V[-1]
            #         - Qu[-(i+1)].T*(Quu[-(i+1)]**(-1))*Qu[-(i+1)]
            #     )

            Vx.append(
                    Qx[-(i+1)]
                    - Qxu[-(i+1)]*(Quu[-(i+1)]**(-1))*Qu[-(i+1)]
                )

            Vxx.append(
                    Qxx[-(i+1)]
                    - Qxu[-(i+1)]*(Quu[-(i+1)]**(-1))*Qux[-(i+1)]
                )

        V = list(reversed(V))
        Vx = list(reversed(Vx))
        Vxx = list(reversed(Vxx))

        dx = [None]*len(Time)
        du = [None]*(len(Time)-1)

        dx[0] = np.matrix([[0],[0],[0],[0]])

        # Forward Pass
        for i in range(len(Time)-1):
            du[i] = -Quu[i]**(-1)*(Qux[i]*dx[i] + Qu[i])
            dx[i+1] = Phi[i]*dx[i] + B[i]*du[i]

        U = np.array(U + np.concatenate(du,axis=0).T)[0]

        IterationNumber+=1

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
    if Animate==True:
        if "U" in kwargs.keys(): del(kwargs["U"]) # prevents multiple instances of U going into animate_trajectory()
        animate_trajectory(Time,X,U,**kwargs)

    if PlotCost==True:
        plt.figure(figsize=(7,5))
        plt.plot(
            list(range(1,len(TrialCosts)+1)),
            TrialCosts,
            marker = 'o')
        plt.gca().set_ylabel("Cost")
        plt.gca().set_xlabel("Iteration Number")
        plt.gca().set_xticks(list(range(1,6*int((len(TrialCosts)+1)/5),int((len(TrialCosts)+1)/5))))
        plt.show()

    return(Time,TotalX,TotalU,TrialCosts)

m1 = 10 # kg
m2 = 1 # kg
L = 0.5 # m
g = 9.8 # m/s²
b1 = 10 # kg/s - Damping coefficient for cart position
b2 = 1 # Nms - Damping coefficient for pendulum angle

d1 = 1 # weight of "Minimize Input Energy" in RunningCost'

c1 = 1 # weight of "Minimize time away from initial position" in RunningCost
c2 = 100 # weight of "Minimize time away from target angle" in RunningCost
c3 = 0 # UNASSIGNED weight of "Minimize time away from target cart velocity" in RunningCost
c4 = 100 # weight of "Minimize time away from target angular velocity" in RunningCost

k1 = 0 # UNASSIGNED weight of "Minimize final position from target position" in TerminalCost
k2 = 100 # weight of 'Minimize final angle from target angle' in TerminalCost
k3 = 0 # UNASSIGNED weight of "Minimize final velocity from target velocity" in TerminalCost
k4 = 100 # weight of  'Minimize final angular velocity from target angular velocity' in TerminalCost

TargetAngle = 0 #in radians
TargetAngularVelocity = 0 #in radians

RunningCost = ["Minimize Input Energy",
                "Minimize time away from target angle",
                "Minimize time away from target angular velocity",
                "Minimize time away from initial position"][:]

TerminalCost = ["Minimize final angle from target angle",
                "Minimize final angular velocity from target angular velocity"][:]
