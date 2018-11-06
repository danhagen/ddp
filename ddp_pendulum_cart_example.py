import numpy as np
import matplotlib.pyplot as plt

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

    X1[0] = 0
    X2[0] = np.pi
    X3[0] = 1
    X4[0] = 0

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

def forward_integrate_dynamics_without_control(ICs,UsingDegrees=True,Animate=False):
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

    dt = 0.0001
    N_seconds = 10
    Time = np.arange(0,N_seconds+dt,dt)
    X1 = np.zeros(len(Time))
    X2 = np.zeros(len(Time))
    X3 = np.zeros(len(Time))
    X4 = np.zeros(len(Time))
    U = np.zeros(len(Time))

    # ICs

    if UsingDegrees:
        X1[0] = ICs[0]
        X2[0] = ICs[1]*(np.pi/180)
        X3[0] = ICs[2]
        X4[0] = ICs[3]*(np.pi/180)
    else:
        X1[0] = ICs[0]
        X2[0] = ICs[1]
        X3[0] = ICs[2]
        X4[0] = ICs[3]

    U[0] = 0

    for i in range(len(Time)-1):
        CurrentX = [X1[i],X2[i],X3[i],X4[i]]
        X1[i+1] = X1[i] + dx1_dt(CurrentX,U[i])*dt
        X2[i+1] = X2[i] + dx2_dt(CurrentX,U[i])*dt
        X3[i+1] = X3[i] + dx3_dt(CurrentX,U[i])*dt
        X4[i+1] = X4[i] + dx4_dt(CurrentX,U[i])*dt

    if Animate:
        animate_trajectory(Time,X1,X2,X3,X4,U)
    else:
        plt.figure(figsize=(15,4))

        ax1 = plt.subplot(221)
        ax1.plot(Time,X1,'r')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cart Position (m)')
        if max(abs(X1-X1[0]))<1e-7:
            ax1.set_ylim([X1[0]-2,X1[0]+2])

        ax2 = plt.subplot(222)
        ax2.plot(Time,X3,'r--')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Cart Velocity (m/s)')
        if max(abs(X3-X3[0]))<1e-7:
            ax2.set_ylim([X3[0]-0.25,X3[0]+0.25])

        ax3 = plt.subplot(223)
        ax3.plot(Time,180*X2/np.pi,'g')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Pendulum Angle (deg)')
        if max(abs(180*X2/np.pi - 180*X2[0]/np.pi))<1e-7:
            ax3.set_ylim([180*X2[0]/np.pi - 5,180*X2[0]/np.pi + 5])

        ax4 = plt.subplot(224)
        ax4.plot(Time,180*X4/np.pi,'g--')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Pendulum Angular \n Velocity (deg/s)')
        if max(abs(X4-X4[0]))<1e-7:
            ax4.set_ylim([X4[0]-1,X4[0]+1])

        plt.show()

def animate_trajectory(Time,X1,X2,X3,X4,U):

        # Angles must be in degrees for animation

    X2d = X2*(180/np.pi)
    X4d = X4*(180/np.pi)


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
                (X1[0]-Cart_Width/2,-Cart_Height/2),
                Cart_Width,
                Cart_Height,
                Color='#4682b4')
    ax1.add_patch(Cart)

    FrontWheel = plt.Circle(
                (X1[0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                Wheel_Radius,
                Color='k')
    ax1.add_patch(FrontWheel)
    FrontWheel_Rivet = plt.Circle(
                (X1[0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                0.2*Wheel_Radius,
                Color='0.70')
    ax1.add_patch(FrontWheel_Rivet)

    BackWheel = plt.Circle(
                (X1[0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                Wheel_Radius,
                Color='k')
    ax1.add_patch(BackWheel)
    BackWheel_Rivet = plt.Circle(
                (X1[0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                0.2*Wheel_Radius,
                Color='0.70')
    ax1.add_patch(BackWheel_Rivet)

    Pendulum, = ax1.plot(
                    [
                        X1[0],
                        X1[0] + Pendulum_Length*np.sin(X2[0])
                    ],
                    [
                        Cart_Height/2 + Pendulum_Width/2,
                        Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(X2[0])
                    ],
                    Color='0.50',
                    lw = 10,
                    solid_capstyle='round'
                    )


    Pendulum_Attachment = plt.Circle((X1[0],Cart_Height/2),2*Pendulum_Width/2,Color='#4682b4')
    ax1.add_patch(Pendulum_Attachment)

    Pendulum_Rivet, = ax1.plot(
        [X1[0]],
        [Cart_Height/2 + Pendulum_Width/2],
        c='k',
        marker='o',
        lw=2
        )

    if max(abs(X1-X1[0]))<1e-7:
        MinimumX = X1[0]-10
        MaximumX = X1[0]+10
    elif max(abs(X1-X1[0]))<2:
        MinimumX = min(X1)-1.25*Cart_Width/2-5
        MaximumX = max(X1)+1.25*Cart_Width/2+5
    else:
        MinimumX = min(X1)-1.25*Cart_Width/2
        MaximumX = max(X1)+1.25*Cart_Width/2
        
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

    Position, = ax2.plot([0],[X1[0]],color = 'g')
    ax2.set_xlim(0,Time[-1])
    ax2.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax2.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X1-X1[0]))<1e-7:
        ax2.set_ylim([X1[0]-2,X1[0]+2])
    else:
        RangeX1 = max(X1)-min(X1)
        ax2.set_ylim([min(X1)-0.1*RangeX1,max(X1)+0.1*RangeX1])

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

    Velocity, = ax4.plot([0],[X3[0]],color='g',linestyle="--")
    ax4.set_xlim(0,Time[-1])
    ax4.set_xticks(list(np.linspace(0,Time[-1],5)))
    ax4.set_xticklabels([str(0),'','','',str(Time[-1])])
    if max(abs(X3-X3[0]))<1e-7:
        ax4.set_ylim([X3[0]-2,X3[0]+2])
    else:
        RangeX3= max(X3)-min(X3)
        ax4.set_ylim([min(X3)-0.1*RangeX3,max(X3)+0.1*RangeX3])
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
        Cart.xy = (X1[i]-Cart_Width/2,-Cart_Height/2)

        FrontWheel.center = (X1[i]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        FrontWheel_Rivet.center = (X1[i]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))

        BackWheel.center = (X1[i]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))
        BackWheel_Rivet.center = (X1[i]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius))

        Pendulum.set_xdata([X1[i],X1[i] + Pendulum_Length*np.sin(X2[i])])
        Pendulum.set_ydata([Cart_Height/2 + Pendulum_Width/2,
                            Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(X2[i])])

        Pendulum_Attachment.center = (X1[i],Cart_Height/2)

        Pendulum_Rivet.set_xdata([X1[i]])

        Position.set_xdata(Time[:i])
        Position.set_ydata(X1[:i])

        Angle.set_xdata(Time[:i])
        Angle.set_ydata(X2d[:i])

        Velocity.set_xdata(Time[:i])
        Velocity.set_ydata(X3[:i])

        AngularVelocity.set_xdata(Time[:i])
        AngularVelocity.set_ydata(X4d[:i])

        return Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Position,Angle,Velocity,AngularVelocity,

    # Init only required for blitting to give a clean slate.
    def init():
        Cart = plt.Rectangle(
                    (X1[0]-Cart_Width/2,-Cart_Height/2),
                    Cart_Width,
                    Cart_Height,
                    Color='#4682b4')
        ax1.add_patch(Cart)

        FrontWheel = plt.Circle(
                    (X1[0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    Wheel_Radius,
                    Color='k')
        ax1.add_patch(FrontWheel)
        FrontWheel_Rivet = plt.Circle(
                    (X1[0]+Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    0.2*Wheel_Radius,
                    Color='0.70')
        ax1.add_patch(FrontWheel_Rivet)

        BackWheel = plt.Circle(
                    (X1[0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    Wheel_Radius,
                    Color='k')
        ax1.add_patch(BackWheel)
        BackWheel_Rivet = plt.Circle(
                    (X1[0]-Cart_Width/4,-(Cart_Height/2 + Wheel_Radius)),
                    0.2*Wheel_Radius,
                    Color='0.70')
        ax1.add_patch(BackWheel_Rivet)

        Pendulum, = ax1.plot(
                        [
                            X1[0],
                            X1[0] + Pendulum_Length*np.sin(X2[0])
                        ],
                        [
                            Cart_Height/2 + Pendulum_Width/2,
                            Cart_Height/2 + Pendulum_Width/2 + Pendulum_Length*np.cos(X2[0])
                        ],
                        Color='0.50',
                        lw = 10,
                        solid_capstyle='round'
                        )

        Pendulum_Attachment = plt.Circle((X1[0],Cart_Height/2),2*Pendulum_Width/2,Color='#4682b4')
        ax1.add_patch(Pendulum_Attachment)

        Pendulum_Rivet, = ax1.plot(
            [X1[0]],
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

        Position, = ax2.plot([0],[X1[0]],color = 'g')

        #Pendulum Angle

        Angle, = ax3.plot([0],[X2d[0]],color = 'r')

        # Cart Velocity

        Velocity, = ax4.plot([0],[X3[0]],color = 'g--')

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
