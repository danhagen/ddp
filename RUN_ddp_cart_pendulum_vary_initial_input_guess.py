from ddp_pendulum_cart_example import *
from mpl_toolkits.mplot3d import Axes3D

N_seconds = 10
dt = 0.1
# U = np.zeros(int(N_seconds/dt))
params = {}
# Perturbation_length = 0.1
InitialAngle = np.arange(5,180,5)[:10]
Perturbation_size = np.arange(0,1100,25)[:20]
# U[:int(Perturbation_length/dt)] = Perturbation_size*np.ones((int(Perturbation_length/dt)))
Time_Constant = np.arange(0.05,1.05,0.025)[:20]

RunningCost = [
        "Minimize Input Energy",
        "Minimize time away from target angle"
    ]

TerminalCost=[
        "Minimize final angle from target angle"
    ]

successValueFunction = return_running_cost_func(
        RunningCost=["Minimize time away from target angle"]
    )

for k in range(len(InitialAngle)):
    for j in range(len(Perturbation_size)):
        for i in range(len(Time_Constant)):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Initial Angle: " + str(InitialAngle[k]))
            print("Time Constant: " + str(Time_Constant[i]))
            print("Perturbation Amplitude: " + str(Perturbation_size[j]))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            U = np.array([
                    Perturbation_size[j]*np.exp(-t/Time_Constant[i])
                for t in np.arange(0,N_seconds,dt)])
            Time,TotalX,TotalU,TrialCosts = cart_pendulum_ddp(
                RunningCost=RunningCost,
                TerminalCost=TerminalCost,
                N_iterations=100,
                N_seconds=N_seconds,
                ICs=[0,InitialAngle[k],0,0],
                dt=dt,
                SaveAsGif=False,
                FileName="ddp_cart_pendulum_10s_RC_input_pos_TC_pos_FROM_50_deg_Exp_input_TC_0p4_Freq_1",
                thresh=1e-5,
                U = U,
                Animate=False,
                PlotCost=False
            )
            successValue = successValueFunction(
                    TotalX[-1][:,int((N_seconds/dt)/2):],
                    TotalU[-1][int((N_seconds/dt)/2):],
                    dt
                )
            if successValue<200:
                print("Successful trial for Time Constant = " + str(Time_Constant[i]))
                # animate_trajectory(Time,TotalX[-1],TotalU[-1])
                params["Trial " + "%02d"%(i+1) + "%02d"%(j+1) + "%02d"%(k+1)] = {
                        "Initial Angle" : InitialAngle[k],
                        "Time Constant" : Time_Constant[i],
                        "Perturbation Amplitude" : Perturbation_size[j],
                        "X" : TotalX[-1],
                        "U" : TotalU[-1],
                        "Success Value" : successValue}


angles = []
time_constants = []
perturbation = []
for key in params.keys():
    angles.append(params[key]['Initial Angle'])
    time_constants.append(params[key]['Time Constant'])
    perturbation.append(params[key]['Perturbation Amplitude'])

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(angles,time_constants,perturbation)
ax.set_title(r"$u(t) = A_\delta \cdot e^{-t/\tau}$",fontsize=16)
ax.set_xlabel(r"$\theta_o$ - Initial Angle (deg)")
ax.set_ylabel(r"$\tau$ - Time Constant (s)")
ax.set_zlabel(r"$A_\delta$ - Perturbation Amplitude (N)")
plt.show()
