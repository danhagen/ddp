import numpy as np
import sympy as sp

gr = sp.Symbol('gr')

b1 = sp.Symbol('b1')
b2 = sp.Symbol('b2')
b3 = sp.Symbol('b3')

m1 = sp.Symbol('m1')
m2 = sp.Symbol('m2')
M = sp.Symbol('M')

L1 = sp.Symbol('L1')
L2 = sp.Symbol('L2')

x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
x3 = sp.Symbol('x3')
x4 = sp.Symbol('x4')
x5 = sp.Symbol('x5')
x6 = sp.Symbol('x6')

u = sp.Symbol('u')

# test = (
#     (
#         L1 * (
#             (M + m1 + m2)
#             - (m1 + m2)*sp.cos(x2)**2
#         )
#         * (
#             m2*(m1 + m2)*(M + m1 + m2)*L1*L2 * (
#                 (m1 + m2)*L1*(x5**2)*sp.sin(x2)
#                 + m2*L2*(x6**2)*sp.sin(x3)
#                 - b1*x4
#                 + u
#             )
#             * (
#                 sp.cos(x3)
#                 - sp.cos(2*x2 - x3)
#             )/2
#             - L1*(m1 + m2)*(M + m1 + m2) * (
#                 m2*L1*L2*(x5**2)*sp.sin(x2 - x3)
#                 + m2*L2*gr*sp.sin(x3)
#                 - b3*x6
#             )
#             * (
#                 (M + m1 + m2)
#                 - (m1 + m2)*sp.cos(x2)**2
#             )
#             + L2*m2*(M + m1 + m2)*(
#                 (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                 - (M + m1 + m2)*sp.cos(x2 - x3)
#             )
#             * (
#                 m2*L1*L2*(x6**2)*sp.sin(x2 - x3)
#                 - (m1 + m2)*L1*gr*sp.sin(x2)
#                 + b2*x5
#             )
#         )
#         + m2*L2*(
#             (M + m1 + m2)*L1*(
#                 (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                 - (M + m1 + m2)*sp.cos(x2 - x3)
#             )
#             * (
#                 m2*L1*L2*(x5**2)*sp.sin(x2 - x3)
#                 + m2*L2*gr*sp.sin(x3)
#                 - b3*x6
#             )
#             * (
#                 (M + m1 + m2)
#                 - (m1 + m2)*sp.cos(x2)**2
#             )
#             - L2*(
#                 L1*(
#                     m2*(M + m1 + m2)*(
#                         (m1 + m2)*sp.cos(x2)*sp.cos(x3) - (M + m1 + m2)*sp.cos(x2 - x3)
#                     )
#                     * (
#                         sp.cos(x3)
#                         - sp.cos(2*x2 - x3)
#                     )
#                     - 2*sp.cos(x2) * (
#                         m2*(
#                             (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                             - (M + m1 + m2)*sp.cos(x2 - x3)
#                         )**2
#                         - (m1 + m2) * (
#                             M + m1 + m2*sp.sin(x3)**2
#                         )
#                         * (
#                             M + m1 + m2
#                             - (m1 + m2)*sp.cos(x2)**2
#                             )
#                     )
#                 )
#                 * (
#                     (m1 + m2)*L1*(x5**2)*sp.sin(x2)
#                     + m2*L2*(x6**2)*sp.sin(x3)
#                     - b1*x4
#                     + u
#                 )
#                 + 2*(M + m1 + m2) * (
#                     M + m1 + m2*sp.sin(x3)**2
#                 )
#                 * (
#                     m2*L1*L2*(x6**2)*sp.sin(x2 - x3)
#                     - (m1 + m2)*L1*gr*sp.sin(x2)
#                     + b2*x5
#                 )
#                 * (
#                     (M + m1 + m2)
#                     - (m1 + m2)*sp.cos(x2)**2
#                 )
#             )/2
#         )
#     )
#     / (
#         m2*L1**2*L2**2 * (
#             m2 * (
#                 (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                 - (M + m1 + m2)*sp.cos(x2 - x3)
#             )**2
#             - (m1 + m2) * (
#                 M + m1 + m2*sp.sin(x3)**2
#             )
#             * (
#                 (M + m1 + m2)
#                 - (m1 + m2)*sp.cos(x2)**2
#             )
#         )
#         * (
#             (M + m1 + m2)
#             - (m1 + m2)*sp.cos(x2)**2
#         )
#     )
# )
#
# test = (
#     (
#         L1 * (
#             (M + m1 + m2)
#             - (m1 + m2)*sp.cos(x2)**2
#         )
#         * (
#             m2*(m1 + m2)*(M + m1 + m2)*L1*L2 * (
#                 (m1 + m2)*L1*(x5**2)*sp.sin(x2)
#                 + m2*L2*(x6**2)*sp.sin(x3)
#                 - b1*x4
#                 + u
#             )
#             * (
#                 sp.cos(x3)
#                 - sp.cos(2*x2 - x3)
#             )/2
#             - L1*(m1 + m2)*(M + m1 + m2) * (
#                 m2*L1*L2*(x5**2)*sp.sin(x2 - x3)
#                 + m2*L2*gr*sp.sin(x3)
#                 - b3*x6
#             )
#             * (
#                 (M + m1 + m2)
#                 - (m1 + m2)*sp.cos(x2)**2
#             )
#             + L2*m2*(M + m1 + m2)*(
#                 (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                 - (M + m1 + m2)*sp.cos(x2 - x3)
#             )
#             * (
#                 m2*L1*L2*(x6**2)*sp.sin(x2 - x3)
#                 - (m1 + m2)*L1*gr*sp.sin(x2)
#                 + b2*x5
#             )
#         )
#         + m2*L2*(
#             (M + m1 + m2)*L1*(
#                 (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                 - (M + m1 + m2)*sp.cos(x2 - x3)
#             )
#             * (
#                 m2*L1*L2*(x5**2)*sp.sin(x2 - x3)
#                 + m2*L2*gr*sp.sin(x3)
#                 - b3*x6
#             )
#             * (
#                 (M + m1 + m2)
#                 - (m1 + m2)*sp.cos(x2)**2
#             )
#             - L2*(
#                 L1*(
#                     m2*(M + m1 + m2)*(
#                         (m1 + m2)*sp.cos(x2)*sp.cos(x3) - (M + m1 + m2)*sp.cos(x2 - x3)
#                     )
#                     * (
#                         sp.cos(x3)
#                         - sp.cos(2*x2 - x3)
#                     )
#                     - 2*sp.cos(x2) * (
#                         m2*(
#                             (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                             - (M + m1 + m2)*sp.cos(x2 - x3)
#                         )**2
#                         - (m1 + m2) * (
#                             M + m1 + m2*sp.sin(x3)**2
#                         )
#                         * (
#                             M + m1 + m2
#                             - (m1 + m2)*sp.cos(x2)**2
#                             )
#                     )
#                 )
#                 * (
#                     (m1 + m2)*L1*(x5**2)*sp.sin(x2)
#                     + m2*L2*(x6**2)*sp.sin(x3)
#                     - b1*x4
#                     + u
#                 )
#                 + 2*(M + m1 + m2) * (
#                     M + m1 + m2*sp.sin(x3)**2
#                 )
#                 * (
#                     m2*L1*L2*(x6**2)*sp.sin(x2 - x3)
#                     - (m1 + m2)*L1*gr*sp.sin(x2)
#                     + b2*x5
#                 )
#                 * (
#                     (M + m1 + m2)
#                     - (m1 + m2)*sp.cos(x2)**2
#                 )
#             )/2
#         )
#     )
#     / (
#         m2*L1**2*L2**2
#     )
# )
#
# test1 = (
#     L1*(M + m1 + m2)*(
#         (
#             L2*m2*sp.sin(x2)*sp.sin(x2 - x3)*(
#                 L1*(m1 + m2)*(
#                     L1*x5**2*(m1 + m2)*sp.sin(x2)
#                     + L2*m2*x6**2*sp.sin(x3)
#                     - b1*x4
#                     + u
#                 )
#                 + (
#                     (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                     - (M + m1 + m2)*sp.cos(x2 - x3)
#                 )
#                 * (
#                     L1*L2*m2*x6**2*sp.sin(x2 - x3)
#                     - L1*gr*(m1 + m2)*sp.sin(x2)
#                     + b2*x5
#                 )
#             )
#         ) / (M + (m1 + m2)*sp.sin(x2)**2)
#         - L1*(m1 + m2)*(
#             L1*L2*m2*x5**2*sp.sin(x2 - x3)
#             + L2*gr*m2*sp.sin(x3)
#             - b3*x6
#         )
#     )
# )
# test2 = (
#     L2*m2*(
#         L1*(M + m1 + m2)*(
#             (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#             - (M + m1 + m2)*sp.cos(x2 - x3)
#         )
#         * (
#             L1*L2*m2*x5**2*sp.sin(x2 - x3)
#             + L2*gr*m2*sp.sin(x3)
#             - b3*x6
#         )
#     )
#     - L2**2*m2*(
#         (
#             L1*(
#                 m2*cos(x3)*(
#                     (m1 + m2)*cos(x2)*cos(x3)
#                     - (M + m1 + m2)*cos(x2 - x3)
#                 )
#                 + (m1 + m2)*cos(x2)*(M + m1 + m2*sin(x3)**2)
#             )
#         )
#         * (
#             L1*x5**2*(m1 + m2)*sp.sin(x2)
#             + L2*m2*x6**2*sp.sin(x3)
#             - b1*x4
#             + u
#         )
#     )
#     - L2**2*m2*(
#         (M + m1 + m2)*(M + m1 + m2*sp.sin(x3)**2)*(
#             L1*L2*m2*x6**2*sp.sin(x2 - x3)
#             - L1*gr*(m1 + m2)*sp.sin(x2)
#             + b2*x5
#         )
#     )
# )
#
# # test_total = test1 + test_2
# test_total = (
#     (
#         L1*(M + (m1 + m2)*sp.sin(x2)**2)*(M + m1 + m2)*sp.sin(x2)*sp.sin(x2 - x3)*(
#             L1*L2*m2*(m1 + m2)
#         )
#         - L1*L2**2*m2*(
#             (
#                 m2*(M + m1 + m2)*sp.sin(x2)*sp.sin(x2 - x3)*(
#                     (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                     - (M + m1 + m2)*sp.cos(x2 - x3)
#                 )
#                 - sp.cos(x2)*(
#                     m2*(
#                         (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                         - (M + m1 + m2)*sp.cos(x2 - x3)
#                     )**2
#                     - (M + (m1 + m2)*sp.sin(x2)**2)*(m1 + m2)*(M + m1 + m2*sp.sin(x3)**2)
#                 )
#             )
#         )
#     ) * (
#         L1*x5**2*(m1 + m2)*sp.sin(x2)
#         + L2*m2*x6**2*sp.sin(x3)
#         - b1*x4
#         + u
#     )
#     + (
#         L1*(M + (m1 + m2)*sp.sin(x2)**2)*(M + m1 + m2)*sp.sin(x2)*sp.sin(x2 - x3)*(
#             L2*m2*((m1 + m2)*sp.cos(x2)*sp.cos(x3) - (M + m1 + m2)*sp.cos(x2 - x3))
#         )
#         - L2**2*m2*(
#             (M + (m1 + m2)*sp.sin(x2)**2)*(M + m1 + m2)*(M + m1 + m2*sp.sin(x3)**2)
#         )
#     ) * (
#         L1*L2*m2*x6**2*sp.sin(x2 - x3)
#         - L1*gr*(m1 + m2)*sp.sin(x2)
#         + b2*x5
#     )
#     + (
#         L1*(M + (m1 + m2)*sp.sin(x2)**2)*(M + m1 + m2)*sp.sin(x2)*sp.sin(x2 - x3)*(
#             -L1*(M + (m1 + m2)*sp.sin(x2)**2)*(m1 + m2)
#         )
#         + L2*m2*(
#             L1*(M + m1 + m2)*(M + (m1 + m2)*sp.sin(x2)**2)*(
#                 (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                 - (M + m1 + m2)*sp.cos(x2 - x3)
#             )
#         )
#     ) * (
#         L1*L2*m2*x5**2*sp.sin(x2 - x3)
#         + L2*gr*m2*sp.sin(x3)
#         - b3*x6
#     )
# )
# test_total_simp_1 = (
#     (
#         L1*L2*m2*(
#             L1*(M + (m1 + m2)*sin(x2)**2)*(m1 + m2)*(M + m1 + m2)*sin(x2)*sin(x2 - x3)
#             - L2*(
#                 m2*((m1 + m2)*cos(x2)*cos(x3) - (M + m1 + m2)*cos(x2 - x3))*(M + m1 + m2)*sin(x2)*sin(x2 - x3)
#                 - cos(x2)*(
#                     m2*(
#                         (m1 + m2)*cos(x2)*cos(x3)
#                         - (M + m1 + m2)*cos(x2 - x3)
#                     )**2
#                     - (M + (m1 + m2)*sin(x2)**2)*(m1 + m2)*(M + m1 + m2*sin(x3)**2)
#                 )
#             )
#         )
#     ) * (
#         L1*x5**2*(m1 + m2)*sp.sin(x2)
#         + L2*m2*x6**2*sp.sin(x3)
#         - b1*x4
#         + u
#     )
#     + (
#         L1*(M + (m1 + m2)*sp.sin(x2)**2)*(M + m1 + m2)*sp.sin(x2)*sp.sin(x2 - x3)*(
#             L2*m2*((m1 + m2)*sp.cos(x2)*sp.cos(x3) - (M + m1 + m2)*sp.cos(x2 - x3))
#         )
#         - L2**2*m2*(
#             (M + (m1 + m2)*sp.sin(x2)**2)*(M + m1 + m2)*(M + m1 + m2*sp.sin(x3)**2)
#         )
#     ) * (
#         L1*L2*m2*x6**2*sp.sin(x2 - x3)
#         - L1*gr*(m1 + m2)*sp.sin(x2)
#         + b2*x5
#     )
#     + (
#         L1*(M + (m1 + m2)*sp.sin(x2)**2)*(M + m1 + m2)*sp.sin(x2)*sp.sin(x2 - x3)*(
#             -L1*(M + (m1 + m2)*sp.sin(x2)**2)*(m1 + m2)
#         )
#         + L2*m2*(
#             L1*(M + m1 + m2)*(M + (m1 + m2)*sp.sin(x2)**2)*(
#                 (m1 + m2)*sp.cos(x2)*sp.cos(x3)
#                 - (M + m1 + m2)*sp.cos(x2 - x3)
#             )
#         )
#     ) * (
#         L1*L2*m2*x5**2*sp.sin(x2 - x3)
#         + L2*gr*m2*sp.sin(x3)
#         - b3*x6
#     )
# )

M_matrix = sp.Matrix(
    [
        [
            4*(m1+m2+M),
            2*(L1*(m1+2*m2)*sp.cos(x2) + L2*m2*sp.cos(x2+x3)),
            2*L2*m2*sp.cos(x2+x3)
        ],
        [
            2*(L1*(m1+2*m2)*sp.cos(x2) + L2*m2*sp.cos(x2+x3)),
            L1**2*(m1+4*m2) + L2**2*m2 + 4*L1*L2*m2*sp.cos(x3),
            L2**2*m2 + 2*L1*L2*m2*sp.cos(x3)
        ],
        [
            2*L2*m2*sp.cos(x2+x3),
            L2**2*m2 + 2*L1*L2*m2*sp.cos(x3),
            L2**2*m2
        ]
    ]
)

b = sp.Matrix(
    [
        [
            2*L1*(m1+2*m2)*(x5**2)*sp.sin(x2)
            + 2*L2*m2*((x5+x6)**2)*sp.sin(x2+x3)
            - 4*b1*x4
            + 4*u
        ],
        [
            2*L1*L2*m2*x6*(2*x5+x6)*sp.sin(x3)
            + 2*L1*(m1+2*m2)*gr*sp.sin(x2)
            + 2*L2*m2*gr*sp.sin(x2+x3)
            - 4*b2*x5
        ],
        [
            -2*L1*L2*m2*(x5**2)*sp.sin(x3)
            + 2*L2*m2*gr*sp.sin(x2+x3)
            - 4*b3*x6
        ]
    ]
)
#
# Xddot = M_matrix**(-1)*b
# print(sp.simplify(Xddot[0,0]))
# print(sp.simplify(Xddot[1,0]))
# print(sp.simplify(Xddot[2,0]))

"""
OUTPUT 1:
def F4([x1,x2,x3,x4,x5,x6],u):
    return(
        -(-L1*(m1 + m2)*(L1*L2*m2*x5**2*np.sin(x2 - x3) + L2*gr*m2*np.sin(x3) - b3*x6)*(M + m1 + m2 - (m1 + m2)*np.cos(x2)**2)*(M*np.cos(x3) - M*np.cos(2*x2 - x3) + m1*np.cos(x3) - m1*np.cos(2*x2 - x3) + m2*np.cos(x3) - m2*np.cos(2*x2 - x3)) + L2*(2*L1*(m2*(m1 + m2)*(M + m1 + m2)*np.sin(x2)**2*np.sin(x2 - x3)**2 - m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))**2 - (m1 + m2)*(M + m1 + m2*np.sin(x3)**2)*(-M - m1 - m2 + (m1 + m2)*np.cos(x2)**2))*(L1*x5**2*(m1 + m2)*np.sin(x2) + L2*m2*x6**2*np.sin(x3) - b1*x4 + U) - (m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))*(-M*np.cos(x3) + M*np.cos(2*x2 - x3) - m1*np.cos(x3) + m1*np.cos(2*x2 - x3) - m2*np.cos(x3) + m2*np.cos(2*x2 - x3)) + 2*(m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))**2 + (m1 + m2)*(M + m1 + m2*np.sin(x3)**2)*(-M - m1 - m2 + (m1 + m2)*np.cos(x2)**2))*np.cos(x2))*(L1*L2*m2*x6**2*np.sin(x2 - x3) - L1*gr*(m1 + m2)*np.sin(x2) + b2*x5)))*(M/2 + m1/2 + m2/2)/(L1*L2*(m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))**2 - (m1 + m2)*(M + m1 + m2*np.sin(x3)**2)*(M + m1 + m2 - (m1 + m2)*np.cos(x2)**2))*(M + m1 + m2)*(M + m1 + m2 - (m1 + m2)*np.cos(x2)**2))
    )
def F5([x1,x2,x3,x4,x5,x6],u):
    return(
        -(L1*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))*(M + m1 + m2)*(L1*L2*m2*x5**2*np.sin(x2 - x3) + L2*gr*m2*np.sin(x3) - b3*x6)*(M + m1 + m2 - (m1 + m2)*np.cos(x2)**2) - L2*(L1*(m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))*(M*np.cos(x3) - M*np.cos(2*x2 - x3) + m1*np.cos(x3) - m1*np.cos(2*x2 - x3) + m2*np.cos(x3) - m2*np.cos(2*x2 - x3)) - 2*(m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))**2 - (m1 + m2)*(M + m1 + m2*np.sin(x3)**2)*(M + m1 + m2 - (m1 + m2)*np.cos(x2)**2))*np.cos(x2))*(L1*x5**2*(m1 + m2)*np.sin(x2) + L2*m2*x6**2*np.sin(x3) - b1*x4 + U) + 2*(M + m1 + m2)*(M + m1 + m2*np.sin(x3)**2)*(L1*L2*m2*x6**2*np.sin(x2 - x3) - L1*gr*(m1 + m2)*np.sin(x2) + b2*x5)*(M + m1 + m2 - (m1 + m2)*np.cos(x2)**2))/2)/(L1**2*L2*(m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))**2 - (m1 + m2)*(M + m1 + m2*np.sin(x3)**2)*(M + m1 + m2 - (m1 + m2)*np.cos(x2)**2))*(M + m1 + m2 - (m1 + m2)*np.cos(x2)**2))
    )
def F6([x1,x2,x3,x4,x5,x6],u):
    return(
        (L1*L2*m2*(m1 + m2)*(L1*x5**2*(m1 + m2)*np.sin(x2) + L2*m2*x6**2*np.sin(x3) - b1*x4 + U)*(M*np.cos(x3) - M*np.cos(2*x2 - x3) + m1*np.cos(x3) - m1*np.cos(2*x2 - x3) + m2*np.cos(x3) - m2*np.cos(2*x2 - x3))/2 + L1*(m1 + m2)*(M + m1 + m2)*(L1*L2*m2*x5**2*np.sin(x2 - x3) + L2*gr*m2*np.sin(x3) - b3*x6)*(-M - m1 - m2 + (m1 + m2)*np.cos(x2)**2) + L2*m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))*(M + m1 + m2)*(L1*L2*m2*x6**2*np.sin(x2 - x3) - L1*gr*(m1 + m2)*np.sin(x2) + b2*x5))/(L1*L2**2*m2*(m2*((m1 + m2)*np.cos(x2)*np.cos(x3) - (M + m1 + m2)*np.cos(x2 - x3))**2 + (m1 + m2)*(M + m1 + m2*np.sin(x3)**2)*(-M - m1 - m2 + (m1 + m2)*np.cos(x2)**2)))
    )

OUTPUT 2:
def F4([x1,x2,x3,x4,x5,x6],u):
    return(
        (L2*(2*(-m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))*(-(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))*(L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3)) + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*np.cos(x2 + x3)) + (-m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))**2 + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(-M - m1 + m2*np.cos(x2 + x3)**2 - m2))*(L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3)))*(L1*L2*m2*x6*(2*x5 + x6)*np.sin(x3) + L1*gr*(m1 + 2*m2)*np.sin(x2) + L2*gr*m2*np.sin(x2 + x3) - 2*b2*x5) - (L1**2*m2*(M + m1 + m2)*(L1*m1*np.cos(x2 - x3) + 4*L1*m2*np.sin(x2)*np.sin(x3) + L2*m1*np.cos(x2) + 2*L2*m2*np.cos(x2) - 2*L2*m2*np.cos(x3)*np.cos(x2 + x3))**2 + (-m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))**2 + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(-M - m1 + m2*np.cos(x2 + x3)**2 - m2))*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(L1*x5**2*(m1 + 2*m2)*np.sin(x2) + L2*m2*(x5 + x6)**2*np.sin(x2 + x3) - 2*b1*x4 + 2*u)) - 2*(-(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))*(L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3)) + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*np.cos(x2 + x3))*((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(L1*L2*m2*x5**2*np.sin(x3) - L2*gr*m2*np.sin(x2 + x3) + 2*b3*x6))/(2*L2*(-m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))**2 + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(-M - m1 + m2*np.cos(x2 + x3)**2 - m2))*((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2)))
    )
def F5([x1,x2,x3,x4,x5,x6],u):
    return(
        (L2*((m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))*((-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))*(L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3)) - ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*np.cos(x2 + x3)) + (-m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))**2 + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(-M - m1 + m2*np.cos(x2 + x3)**2 - m2))*(L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3)))*(L1*x5**2*(m1 + 2*m2)*np.sin(x2) + L2*m2*(x5 + x6)**2*np.sin(x2 + x3) - 2*b1*x4 + 2*u) - 2*((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(M + m1 + m2)*(-M - m1 + m2*np.cos(x2 + x3)**2 - m2)*(L1*L2*m2*x6*(2*x5 + x6)*np.sin(x3) + L1*gr*(m1 + 2*m2)*np.sin(x2) + L2*gr*m2*np.sin(x2 + x3) - 2*b2*x5)) - 2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))*((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(M + m1 + m2)*(L1*L2*m2*x5**2*np.sin(x3) - L2*gr*m2*np.sin(x2 + x3) + 2*b3*x6))/(L2*(-m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))**2 + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(-M - m1 + m2*np.cos(x2 + x3)**2 - m2))*((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2)))
    )
def F6([x1,x2,x3,x4,x5,x6],u):
    return(
        (2*L2*m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))*(M + m1 + m2)*(L1*L2*m2*x6*(2*x5 + x6)*np.sin(x3) + L1*gr*(m1 + 2*m2)*np.sin(x2) + L2*gr*m2*np.sin(x2 + x3) - 2*b2*x5) + L2*m2*(-(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))*(L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3)) + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*np.cos(x2 + x3))*(L1*x5**2*(m1 + 2*m2)*np.sin(x2) + L2*m2*(x5 + x6)**2*np.sin(x2 + x3) - 2*b1*x4 + 2*u) + 2*((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(M + m1 + m2)*(L1*L2*m2*x5**2*np.sin(x3) - L2*gr*m2*np.sin(x2 + x3) + 2*b3*x6))/(L2**2*m2*(-m2*(-(2*L1*np.cos(x3) + L2)*(M + m1 + m2) + (L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))*np.cos(x2 + x3))**2 + ((L1*(m1 + 2*m2)*np.cos(x2) + L2*m2*np.cos(x2 + x3))**2 - (M + m1 + m2)*(L1**2*(m1 + 4*m2) + 4*L1*L2*m2*np.cos(x3) + L2**2*m2))*(-M - m1 + m2*np.cos(x2 + x3)**2 - m2)))
    )
"""
test = (
    m2*L1*L2*(
        L1*(m1 + m2)*(M + m1 + m2)*sp.sin(x2)*sp.sin(x2 - x3)
        + L2*(
            m2*sp.cos(x3)*sp.cos(x2 - x3)*(
                M + m1 + m2 - (m1 + m2)*sp.cos(x2)*sp.cos(x3)
            )
        - (m1 + m2)*sp.cos(x2)*(M + m1 + m2*sp.sin(x3)**2)
        )
    )
    * (
        m2*L2*(x6**2)*sp.sin(x3)
        - b1*x4
        + u
    )
    + L1**2*L2*m2*(
        -L1*M*(m1 + m2)*(M + m1 + m2)*sin(x2 - x3)
        - L2*m2*(M + m1 + m2)**2*sin(2*(x2 - x3))/2
        - L2*m2*(m1 + m2)**2*sin(x2)*cos(x2)*cos(x3)**2*(
            cos(x2 - x3) - 1
        )
        - L2*m2*(m1 + m2)*(M + m1 + m2)*sin(2*x3)/2
        - L2*(m1 + m2)**2*(M + m1 + m2)*sin(2*x2)/2
    )
    * (x5**2)
    + m2*(M + m1 + m2)*L2*(
        L1*(
            (m1 + m2)*sp.cos(x2)*sp.cos(x3)
            - (M + m1 + m2)*sp.cos(x2 - x3)
        )
        - L2*(M + m1 + m2*sp.sin(x3)**2)
    )
    * (
        m2*L1*L2*(x6**2)*sp.sin(x2 - x3)
        - (m1 + m2)*L1*gr*sp.sin(x2)
        + b2*x5
    )
    + (M + m1 + m2)*L1*(
        -L1*(m1 + m2)*(M + (m1 + m2)*sp.sin(x2)**2)
        + L2*m2*(
            (m1 + m2)*sp.cos(x2)*sp.cos(x3)
            - (M + m1 + m2)*sp.cos(x2 - x3)
        )
    )
    * (
        + m2*L2*gr*sp.sin(x3)
        - b3*x6
    )
)
