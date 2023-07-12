#!/usr/bin/python3
import numpy as np
import pulp as pl
import random
import copy as cp


MAX_MOVE_RADIUS = 2.7/111
MAX_L_VALUE = 2
MAX_La_VALUE = 3


# The python code of the GRA and GMRA
class Assignment:
	@classmethod
	def GRA(cls, Q, L):
		row = len(Q)
		col = len(Q[0])
		La = [1]*row
		# build a optimal problem
		pro = pl.LpProblem('Minimized the traveled distance of the workers', pl.LpMinimize)
		# build variables for the optimal problem
		lpvars = [[pl.LpVariable("x"+str(i)+"y"+str(j), lowBound = 0, upBound = 1, cat='Integer') for j in range(col)] for i in range(row)]


		# build optimal function
		all = pl.LpAffineExpression()
		for i in range(0,row):
			for j in range(0,col):
				all += Q[i][j]*lpvars[i][j]

		pro += all

		# build constraint for each role
		for j in range(0,col):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j],1) for i in range(0,row)]) , 0, "L"+str(j), L[j])

		# build constraint for each agent
		for i in range(0,row):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j],1) for j in range(0,col)]) , -1,"La"+str(i), La[i])

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

		return [T, pl.value(pro.status), pl.value(pro.objective), assignment_pairs]

	@classmethod
	def GMRA(cls, Q, L, La):
		row = len(Q)
		col = len(Q[0])
		# La = [1]*row
		# build a optimal problem
		pro = pl.LpProblem('Minimized the traveled distance of the workers', pl.LpMinimize)
		# build variables for the optimal problem
		lpvars = [[pl.LpVariable("x"+str(i)+"y"+str(j), lowBound = 0, upBound = 1, cat='Integer') for j in range(col)] for i in range(row)]


		# build optimal function
		all = pl.LpAffineExpression()
		for i in range(0,row):
			for j in range(0,col):
				all += Q[i][j]*lpvars[i][j]

		pro += all

		# build constraint for each role
		for j in range(0,col):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j],1) for i in range(0,row)]) , 0, "L"+str(j), L[j])

		# build constraint for each agent
		for i in range(0,row):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j],1) for j in range(0,col)]) , -1,"La"+str(i), La[i])

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

		return [T, pl.value(pro.status), pl.value(pro.objective), assignment_pairs]


# Generate a Q matrix
def genQMatrix(row, col):
	QMatrix = np.zeros((row, col))
	for i in range(row):
		for j in range(col):
			QMatrix[i][j] = random.randint(1, 30)
	QMatrix = np.array(QMatrix)
	orig_QMatrix = cp.deepcopy(QMatrix)
	return (QMatrix-QMatrix.min())/(QMatrix.max()-QMatrix.min()), orig_QMatrix



'''
 * This program is to conduct real-world scenario to verify the GMRA model.

 * Created by Qian Jiang, June 30, 2023.
'''


if __name__ == '__main__':
	agent_number = 30
	role_number = 10
	row = agent_number
	col = role_number
	# Initialize the vector L
	L = []
	for j in range(col):
		L.append(random.randint(1, MAX_L_VALUE))
	# Initialize the vector La
	La = []
	for i in range(row):
		La.append(random.randint(1, MAX_La_VALUE))
	# generate a Q matrix
	Q_Matrix, orig_QMatrix = genQMatrix(row, col)
	# Solving the GMRA model
	T_Matrix_GMRA, status_GRMA, objective_value_GMRA, T_pairs_GMRA = Assignment.GMRA(orig_QMatrix, L, La)
	print(L)
	print(La)
	print(status_GRMA)
	print(objective_value_GMRA)
	print(T_pairs_GMRA)