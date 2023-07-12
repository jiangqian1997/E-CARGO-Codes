#!/usr/bin/python3
import numpy as np
import pulp as pl
import copy as cp


# GRACCF model
class Assignment:
	@classmethod
	def GRACCF(cls, Q, L, dimension_relationMat=[]):
		row = len(Q)
		col = len(Q[0])
		len_relationMat = len(dimension_relationMat)
		La = [1]*row
		# build a optimal problem
		pro = pl.LpProblem('Max(connection and coverage)', pl.LpMaximize)
		# build variables for the optimal problem
		lpvars = [[pl.LpVariable("x"+str(i)+"y"+str(j), lowBound = 0, upBound = 1, cat='Integer') for j in range(col)] for i in range(row)]

		lpvars_GRACCF = [pl.LpVariable("val_k"+str(k), lowBound = 0, upBound = 1, cat='Integer') for k in range(len_relationMat)]

		# build optimal function
		all = pl.LpAffineExpression()
		for i in range(0,row):
			for j in range(0,col):
				all += Q[i][j]*lpvars[i][j]
		for k in range(0, len_relationMat):
			all += dimension_relationMat[k][4]*Q[dimension_relationMat[k][0]][dimension_relationMat[k][1]]*lpvars_GRACCF[k]

		pro += all

		# build constraint for each role
		for j in range(0,col):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j],1) for i in range(0,row)]) , 0,"L"+str(j),L[j])

		# build constraint for each agent
		for i in range(0,row):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j],1) for j in range(0,col)]) , -1,"La"+str(i), La[i])


		# Introduce constraints 15-16 from GRACCF
		for k in range(0, len_relationMat):
			pro += lpvars_GRACCF[k]*2 <= lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]]+lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]]

		for k in range(0, len_relationMat):
			pro += lpvars_GRACCF[k]+1 >= lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]]+lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]]

		# solve optimal problem
		status = pro.solve()
		print("Assignment Status: ", pl.LpStatus[status])
		print("Final Assignment Result", pl.value(pro.objective))

		# get the result of T matrix
		T = [[ lpvars[i][j].varValue for j in range(col) ] for i in range(row)]

		# record the assignment pairs of T matrix
		assignment_pairs = []
		for i in range(len(T)):
			for j in range(len(T[0])):
				if T[i][j] != 0:
					assignment_pairs.append((i,j))
		assignment_pairs = sorted(assignment_pairs, key=lambda x: x[1])

		return [T,pl.value(pro.status),pl.value(pro.objective), assignment_pairs]


# generate a random Q matrix
def genRandQMat(agentNum, roleNum):
	resMat = np.zeros((agentNum, roleNum))
	for i in range(agentNum):
		for j in range(roleNum):
			resMat[i][j] = round(np.random.random(), 2)
	return resMat

# Perform dimensionality reduction on the relational matrix
# 对ACC关系矩阵进行降维处理
def dimensionalityReduction(relationMat, correspondingList):
	resMat = []
	for i in range(len(relationMat)):
		for j in range(len(relationMat[0])):
			if relationMat[i][j] != 0:
				resMat.append([int(correspondingList[i][0]), int(correspondingList[i][1]), int(correspondingList[j][0]), int(correspondingList[j][1]), relationMat[i][j]])
	return resMat

# Find the location of the CCF matrix through the relationship of the matrix indices.
# 通过矩阵的下标的关系找到CCF矩阵的位置
def getCCF_index(agent_index, role_index, role_num=4):
	return agent_index*role_num+role_index


# A simple case of the CCF matrix in the paper named GRACCF
# GRACCF论文中的关系矩阵
def recurGRACCF_Mat(matLength):
	resMat = np.zeros((matLength, matLength))
	resMat[1][5]=-0.3;resMat[1][6]=0.35;resMat[1][7]=0.35;resMat[1][20]=-0.4;resMat[1][50]=0.8;resMat[1][51]=0.9;

	resMat[2][5]=-0.2;resMat[2][6]=-0.2;resMat[2][7]=0.2;resMat[2][20]=-0.5;resMat[2][50]=0.5;resMat[2][51]=0.6;

	resMat[5][1]=-0.2;resMat[5][2]=0.2;resMat[5][17]=0.2;resMat[5][18]=0.2;resMat[5][19]=0.3;resMat[5][44]=-0.3;resMat[5][46]=0.35;resMat[5][50]=0.7;resMat[5][51]=0.6;

	resMat[6][1]=-0.35;resMat[6][2]=-0.2;resMat[6][17]=0.2;resMat[6][18]=0.2;resMat[6][19]=0.3;resMat[6][44]=-0.2;resMat[6][46]=-0.2;

	resMat[7][1]=0.35;resMat[7][2]=0.4;resMat[7][17]=0.2;resMat[7][18]=0.2;resMat[7][19]=0.2;resMat[7][44]=-0.3;resMat[6][46]=0.35;

	resMat[17][5]=0.2;resMat[17][6]=0.2;resMat[17][7]=0.3;resMat[17][20]=-0.5;resMat[17][21]=-0.4;resMat[17][23]=-0.3;resMat[17][50]=0.6;resMat[17][51]=0.7;

	resMat[18][5]=0.2;resMat[18][6]=0.2;resMat[18][7]=0.3;resMat[18][20]=-0.4;resMat[18][21]=-0.45;resMat[18][23]=-0.3;

	resMat[19][5]=0.2;resMat[19][6]=0.2;resMat[19][7]=0.2;resMat[19][20]=-0.2;resMat[19][21]=-0.2;resMat[19][23]=-0.3;

	resMat[20][17]=0.3;resMat[20][18]=0.2;resMat[20][19]=0.3;resMat[20][50]=0.8;resMat[20][51]=0.7;


	resMat[21][17]=-0.4;resMat[21][18]=-0.45;resMat[21][19]=-0.2;resMat[21][50]=0.6;resMat[21][51]=0.5;

	resMat[23][17]=-0.2;resMat[23][18]=-0.2;resMat[23][19]=-0.3;

	resMat[44][1]=0.3;resMat[44][2]=0.2;resMat[44][5]=-0.3;resMat[44][6]=0.35;resMat[44][7]=0.35;resMat[44][17]=0.3;resMat[44][18]=0.2;resMat[44][19]=0.1;resMat[44][20]=-0.5;resMat[44][21]=-0.4;
	resMat[44][23]=-0.3;resMat[44][50]=0.7;resMat[44][51]=0.7;

	resMat[46][5]=-0.2;resMat[46][6]=-0.2;resMat[46][7]=0.2;resMat[46][50]=0.6;resMat[46][51]=0.6;

	resMat[50][20]=-0.4;resMat[50][44]=0.8;resMat[50][46]=0.6;

	resMat[51][20]=-0.3;resMat[51][44]=0.8;resMat[51][46]=0.6;

	return resMat



'''
 * This program is to conduct real-world scenario to verify the GRACCF model.

 * Created by Qian Jiang, June 30, 2023.

'''


if __name__ == '__main__':
	positionType = 4
	empolyeeNum = 13
	matLength = positionType * empolyeeNum
	L = [1, 2, 4, 2]
	Q_Matrix = [
		[0.18, 0.82, 0.29, 0.01],
		[0.35, 0.80, 0.58, 0.35],
		[0.84, 0.85, 0.86, 0.36],
		[0.96, 0.51, 0.45, 0.64],
		[0.22, 0.33, 0.68, 0.33],
		[0.96, 0.50, 0.10, 0.73],
		[0.25, 0.18, 0.23, 0.39],
		[0.56, 0.35, 0.80, 0.62],
		[0.49, 0.09, 0.33, 0.58],
		[0.38, 0.54, 0.72, 0.20],
		[0.91, 0.31, 0.34, 0.15],
		[0.85, 0.34, 0.43, 0.18],
		[0.44, 0.06, 0.66, 0.37]
		]
	correspondingList = []
	for i in range(empolyeeNum):
		for j in range(positionType):
			correspondingList.append((i,j))
	# generate the CCF matrix
	relationMat_orig = recurGRACCF_Mat(matLength)
	relationMat = cp.deepcopy(relationMat_orig)
	dimension_relationMat = dimensionalityReduction(relationMat, correspondingList)
	# Solving the GRACCF matrix
	TMatrix, result, performance, assignment_pairs = Assignment.GRACCF(Q_Matrix, L, dimension_relationMat)
	print(performance)
	print(result)
	print(assignment_pairs)