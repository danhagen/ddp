import numpy as np
import matplotlib.pyplot as plt
from danpy.sb import *
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import matplotlib.patches as patches
from params import *

def animate_trajectory(Time,X,U,**kwargs):
    SaveAsGif = kwargs.get("SaveAsGif",False)
    assert type(SaveAsGif)==bool, "SaveAsGif must be either True or False (Default)."

    FileName = kwargs.get("FileName","ddp_cart_pole")
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

    TimeStamp = ax0.text(
        0.75*MinimumX,
        0.75*1.50*(Cart_Height/2+Pendulum_Width/2+Pendulum_Length+Pendulum_Width/2),
        "Time: "+str(Time[0])+" s",
        color='0.50',
        fontsize=16
    )

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

        TimeStamp.set_text("Time: "+str(Time[i])+" s")

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

        return SmallMarkers,Markers,Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Input,Position,Angle,Velocity,AngularVelocity,TimeStamp,

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

        TimeStamp = ax0.text(
            0.75*MaximumX,
            0.75*1.50*(Cart_Height/2+Pendulum_Width/2+Pendulum_Length+Pendulum_Width/2),
            "Time: "+str(Time[0])+" s",
            color='0.50',
            fontsize=16
        )

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
        TimeStamp.set_visible(False)
        Ground.set_visible(True)
        Input.set_visible(False)
        Position.set_visible(False)
        Angle.set_visible(False)
        Velocity.set_visible(False)
        AngularVelocity.set_visible(False)

        return SmallMarkers,Markers,Cart,FrontWheel,FrontWheel_Rivet,BackWheel,BackWheel_Rivet,Pendulum,Pendulum_Attachment,Pendulum_Rivet,Ground,Input,Position,Angle,Velocity,TimeStamp,

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
