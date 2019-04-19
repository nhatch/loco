import numpy as np

# My custom DOF order (some signs are switched wrt DART, and hip Euler order switched):
# X, Y, Z, Yw, P, R
# Right: HY, HP, HR, K, AP, AR
# Left:  HY, HP, HR, K, AP, AR
# ? (head yaw or something)

# DART DOF order:
# X, Z, Y, Yw, P, R
# ?, ?, ?, ?, ?, ?, ?, ? (arms/head)
# Left:  HY, HR, HP, K, AP, AR
# Right: HY, HR, HP, K, AP, AR

# Robot DOF order (motor only; all angles converted to range 0-1023):
# ?, ?, ?, ?, ?, ? (arms)                 [Motor IDs: 1--6]
# R-HY, L-HY, R-HR, L-HR, R-HP, L-HP      [Motor IDs: 7--12]
# R-K,  L-K,  R-AP, L-AP, R-HR, L-HR      [Motor IDs: 13--18]
# ?, ? (head)                             [Motor IDs: 19--20]

def fromRobot(positions):
	# reorder joints
	index=[1,3,5,0,2,4,18,19,7,9,11,13,15,17,6,8,10,12,14,16]
	# convert from int values to radians
	simState = np.zeros(len(positions))
	for i in range(len(positions)):
		simState[i] = (positions[i]-2048)*(np.pi/180)*0.088



	return simState[index]

def toRobot(positions):
	# reorder joints
	index=[3,0,4,1,5,2,14,8,15,9,16,10,17,11,18,12,19,13,6,7]
	# convert from radians to int
	robotState = np.zeros(len(positions))
	for i in range(len(positions)):
		robotState[i] = int(positions[i]*180*(1/(np.pi*0.088))) + 2048

	return robotState[index].astype(int).tolist()

