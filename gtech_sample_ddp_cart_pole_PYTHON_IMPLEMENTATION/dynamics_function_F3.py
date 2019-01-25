import numpy as np
from dynamics_function_F4 import *
from params import *

def F3(X,U):
    """
    Takes in the current state values and the applied force U and returns the derivative of the second var (dx3). 

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] -

    #######################

    NOTES:

    - I have replaced the original dynamics with the complex physical pendulum EOM that I derived. This did not effect the behavior of the system and the input/output looked very similar to the simple pendulum equations. (DAH - 01/24/2019)

    def dx3_dt(X,U):
        return(
            (
                m2*L*np.sin(X[1])*(X[3]**2)
                - b1*X[2]
                - m2*gr*np.sin(2*X[1])/2
                + b2*np.cos(X[1])*X[3]/L
                + U
            )
            /
            (m1 + m2*(np.sin(X[1])**2))
        )

    """

    # pos = X[0];
    # theta= X[1];
    # vel = X[2];
    # ang_vel = X[3];
    # num2 = m2*L*(ang_vel**2*sin(theta)-F4(pos,theta,vel,ang_vel,U)*cos(theta));
    # den = (m1+m2);
    # dx3 = (U+num2)/den;
    # dx3 = (
    #         U
    #         + m2*L*(
    #             (X[3]**2)*np.sin(X[1])
    #             - F4(X,U)*np.cos(X[1])
    #         )
    #     ) / (m1+m2)
    #
    # return(dx3)

    return(
        (
            m2*L*np.sin(X[1])*(X[3]**2)
            - b1*X[2]
            - m2*gr*np.sin(2*X[1])/2
            + b2*np.cos(X[1])*X[3]/L
            + U
        )
        /
        (m1 + m2*(np.sin(X[1])**2))
    )
