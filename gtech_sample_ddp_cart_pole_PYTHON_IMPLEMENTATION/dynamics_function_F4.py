import numpy as np
from params import *

def F4(X,U):
    """
    Takes in the current state values and the applied force U and returns the derivative of the fourth var (dx4).

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] -

    #######################

    NOTES:

    - I have replaced the original dynamics with the complex physical pendulum EOM that I derived. This did not effect the behavior of the system and the input/output looked very similar to the simple pendulum equations. (DAH - 01/24/2019)

    def dx4_dt(X,U):
        return(
            (
                -m2*np.sin(2*X[1])*(X[3]**2)/2
                + b1*np.cos(X[1])*X[2]/L
                + (m1+m2)*gr*np.sin(X[1])/L
                - (m1+m2)*b2*X[3]/(m2*(L**2))
                - np.cos(X[1])*U/L
            )
            /
            (m1 + m2*(np.sin(X[1])**2))
        )
    """
    # theta= X[1];
    # ang_vel = X[3];
    #
    # num1 = gr*sin(theta);
    # num2n = -U-m2*L*ang_vel**2*sin(theta);
    # num2d = m1+m2;
    # num2 = cos(theta)*num2n/num2d;
    # num = num1+num2;
    #
    # den3 = (m2*cos(theta)**2/(m2+m1));
    # den = (L*((4/3)-den3));
    # dx4 = num/den;
    # dx4 = (
    #         gr*np.sin(X[1])
    #         + np.cos(X[1])*((-U-m2*L*(X[3]**2)*np.sin(X[1]))/(m1+m2))
    #     )/(L*(4/3-(m2*(np.cos(X[1])**2))/(m1+m2)))
    #
    # return(dx4)
    return(
        (
            -m2*np.sin(2*X[1])*(X[3]**2)/2
            + b1*np.cos(X[1])*X[2]/L
            + (m1+m2)*gr*np.sin(X[1])/L
            - (m1+m2)*b2*X[3]/(m2*(L**2))
            - np.cos(X[1])*U/L
        )
        /
        (m1 + m2*(np.sin(X[1])**2))
    )
