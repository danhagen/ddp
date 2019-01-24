import numpy as np

def fnCostComputation(x_traj,u_new,p_target,dt,Q_f,R):
    """
    This takes in the state variables over time (x_traj) and the input (u_new), as well as the target state (p_target), the time step (dt), and the cost matrices (Q_f and R), and output the cost of the trial.

    This looks like it is simply integrating u_new.T * R * u_new and then adding (x_traj[:,-1]-p_target).T * Q_f * (x_traj[:,-1]-p_target). If so, then we can get rid of the for loop and the Horizon definition as we can simply find this with:
        Cost = (
            np.trapz(u_new.T * R * u_new,dx=dt)
            + (x_traj[:,-1] - p_target).T * Q_f * (x_traj[:,-1] - p_target)
        )

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - See if this can be done with np.trapz and if it is faster this way.
    [ ] - Create a test to ensure that everything has the right shape.

    """
    [numOfStates,Horizon] = np.shape(x_traj)

    # This looks like a simple integration of u_new.T * R * u_new.
    Cost = 0
    for j in range(Horizon-1):
        Cost = Cost + 0.5 * u_new[j].T * R * u_new[j] * dt

    TerminalCost = (
        (np.matrix(x_traj[:,Horizon-1]).T - p_target).T 
        * Q_f
        * (np.matrix(x_traj[:,Horizon-1]).T - p_target)
    )

    Cost = Cost + TerminalCost
    return(Cost)
