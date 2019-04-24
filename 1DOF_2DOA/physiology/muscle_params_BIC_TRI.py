import numpy as np
from physiology.muscle_settings import *
from physiology.MA_functions import *

AllMuscleSettings = return_muscle_settings(PreselectedMuscles=[5,6])

R_Transpose, dR_Transpose, d2R_Transpose = \
			return_MA_matrix_functions(AllMuscleSettings,ReturnMatrixFunction=False,θ_PS=np.pi/2)


"""
R_Transpose, dR_Transpose, and d2R_Transpose are of the form (n,m), where n is the number of muscles and m in the number of joints. In order to unpack the two muscles used in this model, we first must get the elbow MA functions R_Transpose[:,1], then change to a 1xn matrix (by the transpose), and then change to an array to reduce the ndmin from 2 to 1.
"""

cT = 27.8
kT = 0.0047
LrT = 0.964

β = 1.55
ω = 0.75
ρ = 2.12

V_max = -9.15
cv0 = -5.78
cv1 = 9.18
av0 = -1.53
av1 = 0
av2 = 0
bv = 0.69

FL = lambda l,lo: np.exp(-abs(((l/lo)**β-1)/ω)**ρ)
FV = lambda l,v,lo: np.piecewise(v,[v<=0, v>0],\
	[lambda v: (V_max - v/lo)/(V_max + (cv0 + cv1*(l/lo))*(v/lo)),\
	lambda v: (bv-(av0 + av1*(l/lo) + av2*(l/lo)**2)*(v/lo))/(bv + (v/lo))])

c_1 = 23.0
k_1 = 0.046
Lr1 = 1.17
η = 0.01

##########################################
############## BIC SETTINGS ##############
##########################################

DELTa_Settings = AllMuscleSettings["BIC"]

α1 = unit_conversion(
	return_primary_source(
		DELTa_Settings["Pennation Angle"])) # rads
m1 = unit_conversion(
	return_primary_source(
		DELTa_Settings["Mass"])) # kg

bm1 = 0.01 # kg/s

PCSA1 = unit_conversion(
	return_primary_source(
		DELTa_Settings["PCSA"]))
F_MAX1 = unit_conversion(
	return_primary_source(
		DELTa_Settings["Maximum Isometric Force"]))
L_CE_max_1 = 1.2 # These values must be adjusted (SENSITIVITY ANALYSIS NEEDED!)

lo1 = unit_conversion(
	return_primary_source(
		DELTa_Settings["Optimal Muscle Length"]))
lTo1 = unit_conversion(
	return_primary_source(
		DELTa_Settings["Optimal Tendon Length"]))

# r11 = R_Transpose[0,0]
r12 = R_Transpose[0,1]
# dr11_dθ1 = dR_Transpose[0,0]
dr12_dθ2 = dR_Transpose[0,1]
# d2r11_dθ12 = d2R_Transpose[0,0]
d2r12_dθ22 = d2R_Transpose[0,1]

def R1(X):
	return(r12(X[0])) #
def dR1_dx1(X):
	return(dr12_dθ2(X[0]))
def d2R1_dx12(X):
	return(d2r12_dθ22(X[0]))

##########################################
############## TRI SETTINGS ##############
##########################################

TRI_Settings = AllMuscleSettings["TRI"]

α2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Pennation Angle"])) # rads
m2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Mass"])) # kg

bm2 = 0.01 # kg/s

PCSA2 = unit_conversion(
	return_primary_source(
		TRI_Settings["PCSA"]))
F_MAX2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Maximum Isometric Force"]))
L_CE_max_2 = 1.2 # These values must be adjusted (SENSITIVITY ANALYSIS NEEDED!)

lo2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Optimal Muscle Length"]))
lTo2 = unit_conversion(
	return_primary_source(
		TRI_Settings["Optimal Tendon Length"]))

# r21 = R_Transpose[1,0]
r22 = R_Transpose[1,1]
# dr21_dθ1 = dR_Transpose[1,0]
dr22_dθ2 = dR_Transpose[1,1]
# d2r21_dθ12 = d2R_Transpose[1,0]
d2r22_dθ22 = d2R_Transpose[1,1]

def R2(X):
	return(r22(X[1]))
def dR2_dx1(X):
	return(dr22_dθ2(X[1]))
def d2R2_dx12(X):
	return(d2r22_dθ22(X[1]))

##########################################
##########################################
##########################################
