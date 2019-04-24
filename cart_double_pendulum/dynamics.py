from params import *
from math import sin,cos,pi
"""

X = ⎧ X1 ⟶ position of the cart
    ⎪ X2 ⟶ proximal angle of pendulum
    ⎪ X3 ⟶ distal angle of pendulum (measured from vertical)
    ⎪ X4 ⟶ velocity of the cart
    ⎪ X5 ⟶ proximal angular velocity of pendulum
    ⎩ X6 ⟶ distal angular velocity of pendulum (measured from vertical)

Ẋ = ⎧ Ẋ1 ⟶ F1(X,U)
    ⎪ Ẋ2 ⟶ F2(X,U)
    ⎪ Ẋ3 ⟶ F3(X,U)
    ⎪ Ẋ4 ⟶ F4(X,U)
    ⎪ Ẋ5 ⟶ F5(X,U)
    ⎩ Ẋ6 ⟶ F6(X,U)

"""
def F1(X,U):
    return(X[3])
def F2(X,U):
    return(X[4])
def F3(X,U):
    return(X[5])
def F4(X,U):
    return(
        -(M + m1 + m2) * (
            -(m1 + m2)*(M + m1 + m2)*L1 * (
                m2*L1*L2*(X[4]**2)*np.sin(X[1] - X[2])
                + m2*L2*gr*np.sin(X[2])
                - b3*X[5]
            )
            * (
                (M + m1 + m2)
                - (m1 + m2)*np.cos(X[1])**2
            )
            * (
                np.cos(X[2])
                - np.cos(2*X[1] - X[2])
            )
            + L2*(
                2*L1*(
                    m2*(m1 + m2)*(M + m1 + m2)*np.sin(X[1])**2*np.sin(X[1] - X[2])**2
                    - m2*(
                        (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                        - (M + m1 + m2)*np.cos(X[1] - X[2])
                    )**2
                    + (m1 + m2)*(M + m1 + m2*np.sin(X[2])**2)*(
                        (M + m1 + m2)
                        - (m1 + m2)*np.cos(X[1])**2
                    )
                )
                * (
                    (m1 + m2)*L1*(X[4]**2)*np.sin(X[1])
                    + m2*L2*X[5]**2*np.sin(X[2])
                    - b1*X[3]
                    + U
                )
                - (
                    m2*(M + m1 + m2)*(
                        (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                        - (M + m1 + m2)*np.cos(X[1] - X[2])
                    )
                    * (
                        -np.cos(X[2])
                        + np.cos(2*X[1] - X[2])
                    )
                    + 2*np.cos(X[1]) * (
                        m2*(
                            (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                            - (M + m1 + m2)*np.cos(X[1] - X[2])
                        )**2
                        - (m1 + m2)*(M + m1 + m2*np.sin(X[2])**2) * (
                            (M + m1 + m2)
                            - (m1 + m2)*np.cos(X[1])**2
                        )
                    )
                )
                * (
                    m2*L1*L2*(X[5]**2)*np.sin(X[1] - X[2])
                    - (m1 + m2)*L1*gr*np.sin(X[1])
                    + b2*X[4]
                )
            )
        )/2
        / (
            (M + m1 + m2)*L1*L2*(
                m2*(
                    (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                    - (M + m1 + m2)*np.cos(X[1] - X[2])
                )**2
                - (m1 + m2)*(M + m1 + m2*np.sin(X[2])**2) * (
                    (M + m1 + m2)
                    - (m1 + m2)*np.cos(X[1])**2
                )
            )
            * (
                (M + m1 + m2)
                - (m1 + m2)*np.cos(X[1])**2
            )
        )
    )
def F5(X,U):
    return(
        -m2*L2*(
            (M + m1 + m2)*L1*(
                (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                - (M + m1 + m2)*np.cos(X[1] - X[2])
            )
            * (
                m2*L1*L2*(X[4]**2)*np.sin(X[1] - X[2])
                + m2*L2*gr*np.sin(X[2])
                - b3*X[5]
            )
            + L2*(
                L1*(
                    m2*np.cos(X[1] - X[2])*(
                        M + m1 + m2 - (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                    )*np.cos(X[2])
                    - (m1 + m2)*np.cos(X[1])*(M + m1 + m2*np.sin(X[2])**2)
                )
                * (
                    (m1 + m2)*L1*(X[4]**2)*np.sin(X[1])
                    + m2*L2*(X[5]**2)*np.sin(X[2])
                    - b1*X[3]
                    + U
                )
                - (M + m1 + m2) * (
                    M + m1 + m2*np.sin(X[2])**2
                )
                * (
                    m2*L1*L2*(X[5]**2)*np.sin(X[1] - X[2])
                    - (m1 + m2)*L1*gr*np.sin(X[1])
                    + b2*X[4]
                )
            )
        )
        / (
            m2*L1**2*L2**2*(
                m2 * (
                    (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                    - (M + m1 + m2)*np.cos(X[1] - X[2])
                )**2
                - (m1 + m2)*(M + (m1 + m2)*np.sin(X[1])**2) * (
                    M + m1 + m2*np.sin(X[2])**2
                )
            )
        )
    )
def F6(X,U):
    return(
        L1*(
            m2*(m1 + m2)*(M + m1 + m2)*L1*L2*(
                np.cos(X[2]) - np.cos(2*X[1] - X[2])
            )/2
            * (
                (m1 + m2)*L1*(X[4]**2)*np.sin(X[1])
                + m2*L2*(X[5]**2)*np.sin(X[2])
                - b1*X[3]
                + U
            )
            - L1*(m1 + m2)*(M + m1 + m2)*(M + (m1 + m2)*np.sin(X[1])**2) * (
                m2*L1*L2*(X[4]**2)*np.sin(X[1] - X[2])
                + m2*L2*gr*np.sin(X[2])
                - b3*X[5]
            )
            + L2*m2*(M + m1 + m2)*(
                (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                - (M + m1 + m2)*np.cos(X[1] - X[2])
            )
            * (
                m2*L1*L2*(X[5]**2)*np.sin(X[1] - X[2])
                - (m1 + m2)*L1*gr*np.sin(X[1])
                + b2*X[4]
            )
        )
        / (
            m2*L1**2*L2**2*(
                m2 * (
                    (m1 + m2)*np.cos(X[1])*np.cos(X[2])
                    - (M + m1 + m2)*np.cos(X[1] - X[2])
                )**2
                - (m1 + m2)*(M + (m1 + m2)*np.sin(X[1])**2) * (
                    M + m1 + m2*np.sin(X[2])**2
                )
            )
        )
    )
# def F4(X,U):
#     return(
#         (L2*(2*(-m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))*(-(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))*(L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2])) + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*np.cos(X[1] + X[2])) + (-m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))**2 + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(-M - m1 + m2*np.cos(X[1] + X[2])**2 - m2))*(L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2])))*(L1*L2*m2*X[5]*(2*X[4] + X[5])*np.sin(X[2]) + L1*gr*(m1 + 2*m2)*np.sin(X[1]) + L2*gr*m2*np.sin(X[1] + X[2]) - 2*b2*X[4]) - (L1**2*m2*(M + m1 + m2)*(L1*m1*np.cos(X[1] - X[2]) + 4*L1*m2*np.sin(X[1])*np.sin(X[2]) + L2*m1*np.cos(X[1]) + 2*L2*m2*np.cos(X[1]) - 2*L2*m2*np.cos(X[2])*np.cos(X[1] + X[2]))**2 + (-m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))**2 + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(-M - m1 + m2*np.cos(X[1] + X[2])**2 - m2))*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(L1*X[4]**2*(m1 + 2*m2)*np.sin(X[1]) + L2*m2*(X[4] + X[5])**2*np.sin(X[1] + X[2]) - 2*b1*X[3] + 2*U)) - 2*(-(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))*(L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2])) + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*np.cos(X[1] + X[2]))*((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(L1*L2*m2*X[4]**2*np.sin(X[2]) - L2*gr*m2*np.sin(X[1] + X[2]) + 2*b3*X[5]))/(2*L2*(-m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))**2 + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(-M - m1 + m2*np.cos(X[1] + X[2])**2 - m2))*((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2)))
#     )
# def F5(X,U):
#     return(
#         (L2*((m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))*((-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))*(L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2])) - ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*np.cos(X[1] + X[2])) + (-m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))**2 + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(-M - m1 + m2*np.cos(X[1] + X[2])**2 - m2))*(L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2])))*(L1*X[4]**2*(m1 + 2*m2)*np.sin(X[1]) + L2*m2*(X[4] + X[5])**2*np.sin(X[1] + X[2]) - 2*b1*X[3] + 2*U) - 2*((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(M + m1 + m2)*(-M - m1 + m2*np.cos(X[1] + X[2])**2 - m2)*(L1*L2*m2*X[5]*(2*X[4] + X[5])*np.sin(X[2]) + L1*gr*(m1 + 2*m2)*np.sin(X[1]) + L2*gr*m2*np.sin(X[1] + X[2]) - 2*b2*X[4])) - 2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))*((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(M + m1 + m2)*(L1*L2*m2*X[4]**2*np.sin(X[2]) - L2*gr*m2*np.sin(X[1] + X[2]) + 2*b3*X[5]))/(L2*(-m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))**2 + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(-M - m1 + m2*np.cos(X[1] + X[2])**2 - m2))*((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2)))
#     )
# def F6(X,U):
#     return(
#         (2*L2*m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))*(M + m1 + m2)*(L1*L2*m2*X[5]*(2*X[4] + X[5])*np.sin(X[2]) + L1*gr*(m1 + 2*m2)*np.sin(X[1]) + L2*gr*m2*np.sin(X[1] + X[2]) - 2*b2*X[4]) + L2*m2*(-(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))*(L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2])) + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*np.cos(X[1] + X[2]))*(L1*X[4]**2*(m1 + 2*m2)*np.sin(X[1]) + L2*m2*(X[4] + X[5])**2*np.sin(X[1] + X[2]) - 2*b1*X[3] + 2*U) + 2*((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(M + m1 + m2)*(L1*L2*m2*X[4]**2*np.sin(X[2]) - L2*gr*m2*np.sin(X[1] + X[2]) + 2*b3*X[5]))/(L2**2*m2*(-m2*(-(2*L1*np.cos(X[2]) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))*np.cos(X[1] + X[2]))**2 + ((L1*(m1 + 2*m2)*np.cos(X[1]) + L2*m2*np.cos(X[1] + X[2]))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(X[2]) + L2**2*m2))*(-M - m1 + m2*np.cos(X[1] + X[2])**2 - m2)))
#     )
