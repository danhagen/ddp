import numpy as np
from Function3 import *
from Function4 import *

def fnsimulate(xo,u_new,Horizon,dt):
    """
    Simulates the system by plugging in the initial conditions (xo), the new input (u_new), the Horizon (as a means of creating the length of the iteration), and the timestep (dt). Returns a (4,Horizon) numpy.ndarray of the states variables over time.

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - See if Horizon is really necessary considering the np.shape(u_new)[1] == Horizon.
    """

    # global m  #Changed the initial paramaters to reflect the new problem.
    # global M
    # global l
    # global gr

    x = np.zeros((4,Horizon))
    x[:,0] = xo
    Fx = np.zeros((4,))

    for k in range(Horizon-1):
        Fx[0] = x[2,k]
        Fx[1] = x[3,k]
        """
            Fx[2] = (
                (
                    (
                        l*(x[3,k]**2)*np.sin(x[1,k])
                        - (
                            (
                                gr*np.sin(x[1,k])
                                - np.cos(x[1,k])*(u_new[k] + m*l*(x[3,k]**2)*sin(x[1,k]))
                            )
                            /(l*(1-(m*np.cos(x[1,k])**2)/(M+m)))
                        )*l*np.cos(x[1,k])
                    )*m/(m+M)
                )
                + u_new[k]/(m+M)
            )

            Fx[3] = (
                (
                    gr*np.sin(x[1,k])
                    - np.cos(x[1,k])*(u_new[k] + m*l*(x[3,k]**2)*np.sin(x[1,k]))
                )/(l*(1-(m*np.cos(x[1,k])**2)/(M+m)))
            )
        """
        Fx[2] = Function3(x[0,k],x[1,k],x[2,k],x[3,k],u_new[k])
        Fx[3] = Function4(x[0,k],x[1,k],x[2,k],x[3,k],u_new[k])

        x[:,k+1] = x[:,k] + Fx * dt
    return(x)
