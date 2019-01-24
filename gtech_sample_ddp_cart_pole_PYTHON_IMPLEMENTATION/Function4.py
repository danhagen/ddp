import numpy as np
from params import *

def Function4(x1,x2,x3,x4,F):
    """
    Takes in the current state values and the applied force F (input u) and returns the derivative of the fourth var (dx4).

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - See if globals work here or if we need to define them as a global var function to be imported.
    """
    # global M,m,l,gr
    # theta= x2;
    # ang_vel = x4;
    #
    # num1 = gr*sin(theta);
    # num2n = -F-m*l*ang_vel**2*sin(theta);
    # num2d = M+m;
    # num2 = cos(theta)*num2n/num2d;
    # num = num1+num2;
    #
    # den3 = (m*cos(theta)**2/(m+M));
    # den = (l*((4/3)-den3));
    # dx4 = num/den;
    dx4 = (
            gr*np.sin(x2)
            + np.cos(x2)*((-F-m*l*(x4**2)*np.sin(x2))/(M+m))
        )/(l*(4/3-(m*(np.cos(x2)**2))/(m+M)))

    return(dx4)
