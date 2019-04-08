import numpy as np

def return_cost_for_a_given_trial(X,U,p_target,dt,Q_f,R):
    """
    This takes in the state variables over time (X) and the input (U), as well as the target state (p_target), the time step (dt), and the cost matrices (Q_f and R), and output the cost of the trial.

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Create a test to ensure that everything has the right shape.

    """
    Horizon = np.shape(X)[1]

    RunningCost = 0
    for j in range(Horizon-1):
        RunningCost = RunningCost + 0.5 * U[j].T * R * U[j] * dt

    TerminalCost = (
        (np.matrix(X[:,Horizon-1]).T - p_target).T
        * Q_f
        * (np.matrix(X[:,Horizon-1]).T - p_target)
    )[0,0]

    Cost = RunningCost + TerminalCost
    return(Cost)
