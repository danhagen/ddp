import numpy as np
import matplotlib.pyplot as plt
from danpy.sb import *
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy import signal

"""
Notes:

X1: position of cart
X2: angle of pendulum
X3: velocity of cart
X4: angular velocity of pendulum

This program does not include any friction or damping for either the pendulum or the cart. As such, the equation for the cart effectively is a double integrator when the pendulum is at equilibrium (even unstable). Meaning that a cart given the initial conditions (0,0,v,0) or (0,±π,v,0) will continue indefinitely towards x → (±∞,0,v,0) or x → (±∞,±π,v,0), respectively, depending on the sign of v. To test, set ICs to:

    X[0,0] = 0
    X[1,0] = np.pi
    X[2,0] = 1
    X[3,0] = 0

We can still control the pendulum angle, but the system is not quite realistic enough. Might want to implement more realistic models in the future.
"""

m1 = 10 # kg
m2 = 1 # kg
L = 0.5 # m
I_zz = 0.08333 # kg⋅m²
g = 9.8 # m/s²

def dx1_dt(X,U):
    return(X[2])
def dx2_dt(X,U):
    return(X[3])
def dx3_dt(X,U):
    return(
        (
            (m2*L**2 + I_zz)
                * m2
                * L
                * X[3]**2
                * np.sin(X[1])
            - m2**2
                * L**2
                * g
                * np.sin(X[1])
                * np.cos(X[1])
            + (m2*L**2 + I_zz)
                * U
        )
        /
        (
            (m1+m2)
                * (m2*L**2 + I_zz)
            - m2**2
                * L**2
                * np.cos(X[1])**2
        )
    )
def dx4_dt(X,U):
    return(
        (
            (m1+m2)
                * m2
                * L
                * g
                * np.sin(X[1])
            - m2**2
                * L**2
                * X[3]**2
                * np.sin(X[1])
                * np.cos(X[1])
            - m2
                * L
                * np.cos(X[1])
                * U
        )
        /
        (
            (m1+m2)
                * (m2*L**2 + I_zz)
            - m2**2
                * L**2
                * np.cos(X[1])**2
        )
    )

def forward_integrate_dynamics(
        ICs,
        UsingDegrees=True,
        Animate=False,
        U=None,
        ReturnX=False,
        **kwargs):
    """
    ICs must be a list of floats and/or ints of length 4.

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **kwargs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    UsingDegrees must be a bool. Default is True. If True, then the ICs for pendulum angle and angular velocity can be given in degrees and degrees per second, respectively.

    Animate must be a bool. Default is False. If True, the program will run animate_trajectory().

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Notes:

    X1: position of cart
    X2: angle of pendulum
    X3: velocity of cart
    X4: angular velocity of pendulum

    This program does not include any friction or damping for either the pendulum or the cart. As such, the equation for the cart effectively is a double integrator when the pendulum is at equilibrium (even unstable). Meaning that a cart given the initial conditions (0,0,v,0) or (0,±π,v,0) will continue indefinitely towards x → (±∞,0,v,0) or x → (±∞,±π,v,0), respectively, depending on the sign of v. To test, set ICs to [0,180,1,0] (UsingDegrees set to True).
    """
    assert type(ICs)==list and len(ICs)==4, "ICs must be a list of length 4."
    LocationStrings = ["1st", "2nd", "3rd", "4th"]
    for i in range(4):
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
    X = np.zeros((4,len(Time)))
    if U is None:
        U = np.zeros(len(Time))
    else:
        assert len(U)==len(Time), "U must be the same length as Time."

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

    U[0] = 0

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
            plt.figure(figsize=(15,4))

            ax1 = plt.subplot(221)
            ax1.plot(Time,X[0,:],'r')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Cart Position (m)')
            if max(abs(X[0,:]-X[0,0]))<1e-7:
                ax1.set_ylim([X[0,0]-2,X[0,0]+2])

            ax2 = plt.subplot(222)
            ax2.plot(Time,X[2,:],'r--')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Cart Velocity (m/s)')
            if max(abs(X[2,:]-X[2,0]))<1e-7:
                ax2.set_ylim([X[2,0]-0.25,X[2,0]+0.25])

            ax3 = plt.subplot(223)
            ax3.plot(Time,180*X[1,:]/np.pi,'g')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Pendulum Angle (deg)')
            if max(abs(180*X[1,:]/np.pi - 180*X[1,0]/np.pi))<1e-7:
                ax3.set_ylim([180*X[1,0]/np.pi - 5,180*X[1,0]/np.pi + 5])

            ax4 = plt.subplot(224)
            ax4.plot(Time,180*X[3,:]/np.pi,'g--')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Pendulum Angular \n Velocity (deg/s)')
            if max(abs(X[3,:]-X[3,0]))<1e-7:
                ax4.set_ylim([X[3,0]-1,X[3,0]+1])

            plt.show()

def animate_trajectory(Time,X,U):

        # Angles must be in degrees for animation

    X2d = X[1,:]*(180/np.pi)
    X4d = X[3,:]*(180/np.pi)


    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot2grid((3,4),(0,0),colspan=4) # animation
    ax2 = plt.subplot2grid((3,4),(1,0),colspan=2) # cart position
    ax3 = plt.subplot2grid((3,4),(1,2),colspan=2) # pendulum angle
    ax4 = plt.subplot2grid((3,4),(2,0),colspan=2) # cart velocty
    ax5 = plt.subplot2grid((3,4),(2,2),colspan=2) # pendulum angular velocity

    plt.suptitle("Cart-Pendulum Example",Fontsize=28,y=0.95)

    Pendulum_Width = 0.5*L
    Pendulum_Length = 2*L
    Cart_Width = 4*L
    Cart_Height = 2*L
    Wheel_Radius = 0.125*Cart_Width
    # Model Drawing

    Cart = plt.Rectangle(
                (X[0,0]-Cart_Width/2,-Cart_Height/2),
                Cart_Width,
                Cart_Height,
                Color='#4682b4')
    ax1.add_patch(Cart)

    FrontWheel = plt.Circle(
                (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                Wheel_Radius,
                Color='k')
    ax1.add_patch(FrontWheel)
    FrontWheel_Rivet = plt.Circle(
                (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                0.2*Wheel_Radius,
                Color='0.70')
    ax1.add_patch(FrontWheel_Rivet)

    BackWheel = plt.Circle(
                (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                Wheel_Radius,
                Color='k')
    ax1.add_patch(BackWheel)
    BackWheel_Rivet = plt.Circle(
                (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                0.2*Wheel_Radius,
                Color='0.70')
    ax1.add_patch(BackWheel_Rivet)

    Pendulum, = ax1.plot(
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


    Pendulum_Attachment = plt.Circle((X[0,0],Cart_Height/2),2*Pendulum_Width/2,Color='#4682b4')
    ax1.add_patch(Pendulum_Attachment)

    Pendulum_Rivet, = ax1.plot(
        [X[0,0]],
        [Cart_Height/2 + Pendulum_Width/2],
        c='k',
        marker='o',
        lw=2
        )

    if max(abs(X[0,:]-X[0,0]))<1e-7:
        MinimumX = X[0,0]-10
        MaximumX = X[0,0]+10
    elif max(abs(X[0,:]-X[0,0]))<2:
        MinimumX = min(X[0,:])-1.25*Cart_Width/2-5
        MaximumX = max(X[0,:])+1.25*Cart_Width/2+5
    else:
        MinimumX = min(X[0,:])-1.25*Cart_Width/2
        MaximumX = max(X[0,:])+1.25*Cart_Width/2

    Ground = plt.Rectangle(
                (MinimumX,-1.50*(Cart_Height/2 + Wheel_Radius*2)),
                MaximumX-MinimumX,
                0.50*(Cart_Height/2 + Wheel_Radius*2),
                Color='0.70')
    ax1.add_patch(Ground)

    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax1.set_frame_on(True)

    ax1.set_xlim([MinimumX,MaximumX])
    ax1.set_ylim(
        [
            -1.50*(Cart_Height/2 + Wheel_Radius*2),
            1.10*(Cart_Height/2+Pendulum_Width/2+Pendulum_Length+Pendulum_Width/2)
        ]
        )
    ax1.set_aspect('equal')

    #Cart Position

    Position, = ax2.plot([0],[X[0,0]],color = 'g')
    ax2.set_xlim(0,Time[-1])
    ax2.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax2.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X[0,:]-X[0,0]))<1e-7:
        ax2.set_ylim([X[0,0]-2,X[0,0]+2])
    else:
        RangeX1 = max(X[0,:])-min(X[0,:])
        ax2.set_ylim([min(X[0,:])-0.1*RangeX1,max(X[0,:])+0.1*RangeX1])

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title("Cart Position (m)",fontsize=16,fontweight = 4,color = 'g',y = 0.95)

    #Pendulum Angle

    Angle, = ax3.plot([0],[X2d[0]],color = 'r')
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
    ax3.set_title("Pendulum Angle (deg)",fontsize=16,fontweight = 4,color = 'r',y = 0.95)

    # Cart Velocity

    Velocity, = ax4.plot([0],[X[2,0]],color='g',linestyle="--")
    ax4.set_xlim(0,Time[-1])
    ax4.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax4.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X[2,:]-X[2,0]))<1e-7:
        ax4.set_ylim([X[2,0]-2,X[2,0]+2])
    else:
        RangeX3= max(X[2,:])-min(X[2,:])
        ax4.set_ylim([min(X[2,:])-0.1*RangeX3,max(X[2,:])+0.1*RangeX3])
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_title("Cart Velocity (m/s)",fontsize=16,fontweight = 4,color = 'g',y = 0.95)

    # Angular Velocity
    AngularVelocity, = ax5.plot([0],[X4d[0]],color='r',linestyle='--')
    ax5.set_xlim(0,Time[-1])
    ax5.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax5.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X4d-X4d[0]))<1e-7:
        ax5.set_ylim([X4d[0]-2,X4d[0]+2])
    else:
        RangeX4d= max(X4d)-min(X4d)
        ax5.set_ylim([min(X4d)-0.1*RangeX4d,max(X4d)+0.1*RangeX4d])
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.set_title("Pendulum Angular Velocity (deg/s)",fontsize=16,fontweight = 4,color = 'r',y = 0.95)

    def animate(i):
        Cart.xy = (X[0,i]-Cart_Width/2,-Cart_Height/2)

        FrontWheel.center = (X[0,i]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        FrontWheel_Rivet.center = (X[0,i]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))

        BackWheel.center = (X[0,i]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        BackWheel_Rivet.center = (X[0,i]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))

        Pendulum.set_xdata([X[0,i],X[0,i] + Pendulum_Length*np.sin(X[1,i])])
        Pendulum.set_ydata([Cart_Height/2 + Pendulum_Width/2,
                            Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(X[1,i])])

        Pendulum_Attachment.center = (X[0,i],Cart_Height/2)

        Pendulum_Rivet.set_xdata([X[0,i]])

        Position.set_xdata(Time[:i])
        Position.set_ydata(X[0,:i])

        Angle.set_xdata(Time[:i])
        Angle.set_ydata(X2d[:i])

        Velocity.set_xdata(Time[:i])
        Velocity.set_ydata(X[2,:i])

        AngularVelocity.set_xdata(Time[:i])
        AngularVelocity.set_ydata(X4d[:i])

        return Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Position,Angle,Velocity,AngularVelocity,

    # Init only required for blitting to give a clean slate.
    def init():
        Cart = plt.Rectangle(
                    (X[0,0]-Cart_Width/2,-Cart_Height/2),
                    Cart_Width,
                    Cart_Height,
                    Color='#4682b4')
        ax1.add_patch(Cart)

        FrontWheel = plt.Circle(
                    (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    Wheel_Radius,
                    Color='k')
        ax1.add_patch(FrontWheel)
        FrontWheel_Rivet = plt.Circle(
                    (X[0,0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    0.2*Wheel_Radius,
                    Color='0.70')
        ax1.add_patch(FrontWheel_Rivet)

        BackWheel = plt.Circle(
                    (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    Wheel_Radius,
                    Color='k')
        ax1.add_patch(BackWheel)
        BackWheel_Rivet = plt.Circle(
                    (X[0,0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    0.2*Wheel_Radius,
                    Color='0.70')
        ax1.add_patch(BackWheel_Rivet)

        Pendulum, = ax1.plot(
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

        Pendulum_Attachment = plt.Circle((X[0,0],Cart_Height/2),2*Pendulum_Width/2,Color='#4682b4')
        ax1.add_patch(Pendulum_Attachment)

        Pendulum_Rivet, = ax1.plot(
            [X[0,0]],
            [Cart_Height/2 + Pendulum_Width/2],
            c='k',
            marker='o',
            lw=2
            )

        Ground = plt.Rectangle(
                    (MinimumX,-1.50*(Cart_Height/2 + Wheel_Radius*2)),
                    MaximumX-MinimumX,
                    0.50*(Cart_Height/2 + Wheel_Radius*2),
                    Color='0.70')
        ax1.add_patch(Ground)

        #Cart Position

        Position, = ax2.plot([0],[X[0,0]],color = 'g')

        #Pendulum Angle

        Angle, = ax3.plot([0],[X2d[0]],color = 'r')

        # Cart Velocity

        Velocity, = ax4.plot([0],[X[2,0]],color = 'g--')

        # Angular Velocity
        AngularVelocity, = ax5.plot([0],[X4d[0]],color = 'r')

        Cart.set_visible(False)
        FrontWheel.set_visible(False)
        FrontWheel_Rivet.set_visible(False)
        BackWheel.set_visible(False)
        BackWheel_Rivet.set_visible(False)
        Pendulum.set_visible(False)
        Pendulum_Attachment.set_visible(False)
        Pendulum_Rivet.set_visible(False)
        Ground.set_visible(True)
        Position.set_visible(False)
        Angle.set_visible(False)
        Velocity.set_visible(False)
        AngularVelocity.set_visible(False)

        return Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Ground,Position,Angle,Velocity,

    dt = Time[1]-Time[0]
    ani = animation.FuncAnimation(fig, animate, frames=np.arange(1, len(Time),2000), init_func=init, blit=False)
    # if save_as_gif:
    # 	ani.save('test.gif', writer='imagemagick', fps=30)
    plt.show()

def fx_32(X,U):
    return(
        (
            (
                (m2*L**2 + I_zz)
                    * m2
                    * L
                    * X[3]**2
                    * np.cos(X[1])
                - m2**2
                    * L**2
                    * g
                    * np.cos(2*X[1])
            )
                * (
                    (m1+m2)
                        * (m2*L**2 + I_zz)
                    - m2**2
                        * L**2
                        * np.cos(X[1])**2
                )
            - (
                (m2*L**2 + I_zz)
                    * m2
                    * L
                    * X[3]**2
                    * np.sin(X[1])
                - m2**2
                    * L**2
                    * g
                    * np.sin(X[1])
                    * np.cos(X[1])
                + (m2*L**2 + I_zz)
                    * U
            )
                * (
                    m2**2
                        * L**2
                        * np.sin(2*X[1])
                )
        )
        /
        (
            (m1+m2)
                * (m2*L**2 + I_zz)
            - m2**2
                * L**2
                * np.cos(X[1])**2
        )**2
    )
def fx_34(X,U):
    return(
        (
            2
                * (m2*L**2 + I_zz)
                * m2
                * L
                * X[3]
                * np.sin(X[1])
        )
        /
        (
            (m1+m2)
                * (m2*L**2 + I_zz)
            - m2**2
                * L**2
                * np.cos(X[1])**2
        )
    )
def fx_42(X,U):
    return(
        (
            (
                (m1+m2)
                    * m2
                    * L
                    * g
                    * np.cos(X[1])
                - m2**2
                    * L**2
                    * X[3]**2
                    * np.cos(2*X[1])
                + m2
                    * L
                    * np.sin(X[1])
                    * U
            )
                * (
                    (m1+m2)
                        * (m2*L**2 + I_zz)
                    - m2**2
                        * L**2
                        * np.cos(X[1])**2
                )
            - (
                (m1+m2)
                    * m2
                    * L
                    * g
                    * np.sin(X[1])
                - m2**2
                    * L**2
                    * X[3]**2
                    * np.sin(X[1])
                    * np.cos(X[1])
                - m2
                    * L
                    * np.cos(X[1])
                    * U
            )
                * (
                    m2**2
                        * L**2
                        * np.sin(2*X[1])
                )
        )
        /
        (
            (m1+m2)
                * (m2*L**2 + I_zz)
            - m2**2
                * L**2
                * np.cos(X[1])**2
        )**2
    )
def fx_44(X,U):
    return(
        (
            -m2**2
                * L**2
                * X[3]
                * np.sin(2*X[1])
        )
        /
        (
            (m1+m2)
                * (m2*L**2 + I_zz)
            - m2**2
                * L**2
                * np.cos(X[1])**2
        )
    )
def fx(X,U):
    return(
        np.matrix(
            [
                [0,0,1,0],
                [0,0,0,1],
                [0,fx_32(X,U),0,fx_34(X,U)],
                [0,fx_42(X,U),0,fx_44(X,U)]
            ]
        )
    )

def fu_31(X,U):
    return(
        (
            m2*L**2 + I_zz
        )
        /
        (
            (m1+m2)
                * (m2*L**2 + I_zz)
            - m2**2
                * L**2
                * np.cos(X[1])**2
        )
    )
def fu_41(X,U):
    return(
        (
            - m2
                * L
                * np.cos(X[1])
        )
        /
        (
            (m1+m2)
                * (m2*L**2 + I_zz)
            - m2**2
                * L**2
                * np.cos(X[1])**2
        )
    )
def fu(X,U):
    return(
        np.matrix(
            [
                [0],
                [0],
                [fu_31(X,U)],
                [fu_41(X,U)]
            ]
        )
    )

def Phi(X,U,dt):
    return(np.eye(4) + fx(X,U)*dt)
def B(X,U,dt):
    return(fu(X,U)*dt)

def l(X,U,dt):
    return((1/2)*U**2*dt)

def find_terminal_cost(X):
    return((1/2)*X[1]**2)

def Q(X,U,V_prime,dt):
    return(l(X,U,dt)+V_Prime)
def return_Qx(X,U,Vx_prime,dt):
    """
    Vx_prime must be a (4,1) array.

    lx is zero for this system.

    Returns a 4x1 array
    """
    return(np.zeros((4,1)) + (Phi(X,U,dt).T)*(Vx_prime[:,np.newaxis]))
def return_Qu(X,U,Vx_prime,dt):
    """
    Vx_prime must be a (4,1) array.

    lu is d/dU[0.5*U**2*dt] = U*dt

    Returns a 1x1 array
    """
    return(U*dt + (B(X,U,dt).T)*(Vx_prime[:,np.newaxis]))
def return_Qux(X,U,Vxx_prime,dt):
    """
    Vxx_prime must be a (4,4) array/matrix.

    lux is zero for this system.

    Returns a 1x4 array
    """
    return(np.zeros((1,4)) + (B(X,U,dt).T)*Vxx_prime*Phi(X,U,dt))
def return_Qxu(X,U,Vxx_prime,dt):
    """
    Vxx_prime must be a (4,4) array/matrix.

    lxu is zero for this system.

    Returns a 4x1 array
    """
    return(np.zeros((4,1)) + (Phi(X,U,dt).T)*Vxx_prime*B(X,U,dt))
def return_Qxx(X,U,Vxx_prime,dt):
    """
    Vxx_prime must be a (4,4) array/matrix.

    lxx is zero for this system.

    Returns a 4x4 array
    """
    return(np.zeros((4,4)) + (Phi(X,U,dt).T)*Vxx_prime*Phi(X,U,dt))
def return_Quu(X,U,Vxx_prime,dt):
    """
    Vxx_prime must be a (4,4) array/matrix.

    luu is dt for this system. d/dU[d/dU[0.5*U**2*dt]] = d/dU[U*dt] = dt

    Returns a 1x1 array
    """
    return(dt + (B(X,U,dt).T)*Vxx_prime*B(X,U,dt))

def return_du_star(dx,X,U,Vx_prime,Vxx_prime,dt):
    return(
        -(return_Quu(X,U,Vxx_prime,dt)**(-1))
            *(
                return_Qux(X,U,Vxx_prime,dt)
                    *(dx[:,np.newaxis])
                + return_Qu(X,U,Vx_prime,dt)
            )
    )

def backward_pass(X,U,V_prime,Vx_prime,Vxx_prime,dt):
    Qu = return_Qu(X,U,Vx_prime,dt)
    Quu_inv = (return_Quu(X,U,Vxx_prime,dt))**(-1)
    Qxu = return_Qxu(X,U,Vxx_prime,dt)
    Qux = Qxu.T

    V = l(X,U,dt) + V_prime - (1/2)*Qu.T*(Quu_inv)*Qu
    Vx = np.array((return_Qx(X,U,Vx_prime,dt) - Qxu*(Quu_inv)*Qu).T)[0,:]
    Vxx = return_Qxx(X,U,Vxx_prime,dt) - Qxu*(Quu_inv)*Qux
    return(V,Vx,Vxx)

def ddp():
    dt = 0.001
    N_seconds = 10
    Time = np.arange(0,N_seconds+dt,dt)
    N_trials = 5

    ICs = [0,10,0,0]
    U = 10*np.sin(np.pi*2*Time) # first set of inputs
    TrialCosts = np.zeros(N_trials)

    TrialNumber = 1
    while TrialNumber <= N_trials:
        print("Trial Number " + str(TrialNumber) + "/" + str(N_trials))
        X = forward_integrate_dynamics(
                ICs,
                U=U,
                ReturnX=True,
                dt=dt,
                N_seconds=N_seconds
            )

        V = np.zeros(len(Time))
        Vx = np.zeros((4,len(Time)))
        Vxx = np.zeros((4,4,len(Time)))

        V[-1] = (1/2)*X[1,-1]**2 + (1/2)*X[3,-1]**2
        Vx[:,-1] = np.array([0,X[1,-1],0,X[3,-1]])
        Vxx[:,:,-1] = np.matrix(
                            [
                            [0,0,0,0],
                            [0,1,0,0],
                            [0,0,0,0],
                            [0,0,0,1]
                            ]
                        )
        statusbar1 = dsb(0,len(Time)-1,title="Backward Pass")
        for i in range(len(Time)-1):
            [V[-(i+1)],Vx[:,-(i+1)],Vxx[:,:,-(i+1)]] = backward_pass(
                                                                X[:,-(i+1)],
                                                                U[-(i+1)],
                                                                V[-i],
                                                                Vx[:,-i],
                                                                Vxx[:,:,-i],
                                                                dt
                                                                )
            statusbar1.update(i)

        dx = np.zeros((4,len(Time)))
        du = np.zeros(len(Time))
        du[0] = return_du_star(dx[:,0],X[:,0],U[0],Vx[:,1],Vxx[:,:,1],dt)
        statusbar2 = dsb(0,len(Time)-1,title="Forward Pass")
        for i in range(len(Time)-1):
            dx[:,i+1] = np.array(
                        (
                            Phi(X[:,0],U[0],dt)*(dx[:,i][:,np.newaxis])
                            + B(X[:,0],U[0],dt)*du[i]
                        ).T
                    )
            if i < len(Time)-2:
                du[i+1] = return_du_star(
                        dx[:,i+1],
                        X[:,i+1],
                        U[i+1],
                        Vx[:,i+2],
                        Vxx[:,:,i+2],
                        dt
                        )
            statusbar2.update(i)

        TrialCosts[TrialNumber-1] = (
               (1/2)*X[1,-1]**2
               + (1/2)*X[3,-1]**2
               + np.trapz((1/2)*U**2,dx=dt)
               )
        import ipdb; ipdb.set_trace()
        U = U + du
        TrialNumber+=1
    return(X,U,TrialCosts)
