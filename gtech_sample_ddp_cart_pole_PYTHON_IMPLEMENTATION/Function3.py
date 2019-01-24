import numpy as np
from Function4 import *
from params import *

def Function3(x1,x2,x3,x4,F):
    """
    Takes in the current state values and the applied force F (input u) and returns the derivative of the second var (dx3).

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Figure out which EOM this reflects as dx2 is vague. It has been changed to dx3.
    [ ] - Test to see if we can use globals in separate scripts.
    """
    # global M,m,l
    # pos = x1;
    # theta= x2;
    # vel = x3;
    # ang_vel = x4;
    # num2 = m*l*(ang_vel**2*sin(theta)-Function4(pos,theta,vel,ang_vel,F)*cos(theta));
    # den = (M+m);
    # dx3 = (F+num2)/den;

    dx3 = (
            F
            + m*l*(
                (x4**2)*np.sin(x2)
                - Function4(x1,x2,x3,x4,F)*np.cos(x2)
            )
        ) / (m+M)

    return(dx3)
