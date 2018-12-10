from ddp_pendulum_cart_example import *

N_seconds = 10
dt = 0.1
# U = np.zeros(int(N_seconds/dt))
params = {}
# Perturbation_length = 0.1
InitialAngle = np.arange(5,180,5)
Perturbation_size = np.arange(0,1100,50)
# U[:int(Perturbation_length/dt)] = Perturbation_size*np.ones((int(Perturbation_length/dt)))
Time_Constant = np.arange(0.05,1.05,0.05)

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
            if successValue<1000:
                print("Successful trial for Time Constant = " + str(Time_Constant[i]))
                # animate_trajectory(Time,TotalX[-1],TotalU[-1])
                params["Trial " + str(i+1)] = {
                        "Initial Angle" : InitialAngle[k],
                        "Time Constant" : Time_Constant[i],
                        "Perturbation Amplitude" : Perturbation_size[j]
                        }
