#!/usr/bin/python3
import numpy as np
import pulp as pl
import copy as cp
import random
import time
import math
import itertools
from gurobipy import *

# MAX_L_VALUE = 8
MAX_La_VALUE = 3


# GRAMAC model
class Assignment:
	@classmethod
	def GRAMAC_latest_gurobi(cls, Q_bar, Q, L, dimension_relationMat=[], res=[]):
		row = len(Q)
		col = len(Q[0])
		len_relationMat = len(dimension_relationMat)
		La = [1] * row
		lambda_value = Q_bar.max() * sum(L)
		permutations_number = len(list(itertools.permutations(range(sum(L)), 2)))

		# Create a new model
		m = Model("GRAMAC")

		# Create variables
		lpvars = [[m.addVar(vtype=GRB.BINARY, name="x"+str(i)+"y"+str(j)) for j in range(col)] for i in range(row)]
		lpvars_GRAMAC = [m.addVar(vtype=GRB.BINARY, name="val"+str(k)) for k in range(len_relationMat)]

		# Set objective
		obj = quicksum(Q_bar[i][j] * lpvars[i][j] for i in range(row) for j in range(col)) + \
			quicksum(lambda_value * dimension_relationMat[k][4] * lpvars_GRAMAC[k] for k in range(len_relationMat))
		m.setObjective(obj, GRB.MINIMIZE)
		# m.setParam(GRB.Param.TimeLimit, 1000)
		# 设置绝对MIP间隙
		m.setParam(GRB.Param.MIPGapAbs, lambda_value)
		# m.setParam(GRB.Param.MIPGapAbs, 2*lambda_value)
		# m.setParam(GRB.Param.MIPGap, 0.1)
		# Suppress the output
		# m.setParam('OutputFlag', 0)
		m.setParam("Threads", 24)
		m.setParam("OutputFlag", 0)


		# Constraints
		for j in range(col):
			m.addConstr(quicksum(lpvars[i][j] for i in range(row)) == L[j], "L"+str(j))

		for i in range(row):
			m.addConstr(quicksum(lpvars[i][j] for j in range(col)) <= La[i], "La"+str(i))

		for k in range(len_relationMat):
			m.addConstr(lpvars_GRAMAC[k] * 2 <= lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]] + lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]])

			m.addConstr(lpvars_GRAMAC[k] + 1 >= lpvars[dimension_relationMat[k][0]][dimension_relationMat[k][1]] + lpvars[dimension_relationMat[k][2]][dimension_relationMat[k][3]])

		# m.addConstr(quicksum(lpvars_GRAMAC[k] for k in range(len(lpvars_GRAMAC))) == permutations_number / 2)

		if res:
			T_greedy, T_greedy_GRAMAC = res
			# 设置初始值
			for i in range(row):
				for j in range(col):
					lpvars[i][j].setAttr(GRB.Attr.Start, T_greedy[i][j])
			for k in range(len_relationMat):
				lpvars_GRAMAC[k].setAttr(GRB.Attr.Start, T_greedy_GRAMAC[k])

		# Optimize model
		m.optimize()

		if m.status == GRB.Status.INFEASIBLE:
			return [[], -1, -1, []]


		# Get the result
		T = [[lpvars[i][j].X for j in range(col)] for i in range(row)]
		T_GRAMAC = [lpvars_GRAMAC[k].X for k in range(len(lpvars_GRAMAC))]
		group_performance = sum(Q[i][j] * lpvars[i][j].X for i in range(row) for j in range(col))

		assignment_pairs = [(i, j) for i in range(row) for j in range(col) if T[i][j] != 0]
		assignment_agents = [i for i in range(row) for j in range(col) if T[i][j] != 0]
		assignment_pairs = sorted(assignment_pairs, key=lambda x: x[1])

		print("Assignment Status: ", m.status)
		print("Final Assignment Result", group_performance)

		return [T, m.status, group_performance, assignment_pairs, assignment_agents]

	@classmethod
	def GRACAG_gurobi(cls, Q, L, A_c):
		row = len(Q)
		col = len(Q[0])
		La = [1] * row

		# Initialize a new model
		m = Model("Maximized the overall group performance")

		# Create variables for the model
		lpvars = [[m.addVar(vtype=GRB.BINARY, name="x"+str(i)+"y"+str(j)) for j in range(col)] for i in range(row)]

		# Set the objective function
		m.setObjective(sum(Q[i][j] * lpvars[i][j] for i in range(row) for j in range(col)), GRB.MAXIMIZE)

		# Suppress the output
		m.setParam('OutputFlag', 0)

		# Add constraints for each role
		for j in range(col):
			m.addConstr(sum(lpvars[i][j] for i in range(row)) == L[j], name="L"+str(j))

		# Add constraints for each agent
		for i in range(row):
			m.addConstr(sum(lpvars[i][j] for j in range(col)) <= La[i], name="La"+str(i))

		# Add agent conflict constraints
		for i in range(row):
			for i1 in range(i, row):
				for j in range(col):
					for j1 in range(col):
						if i != i1:
							m.addConstr(A_c[i][i1] * (lpvars[i][j] + lpvars[i1][j1]) <= 1)

		# Optimize the model
		m.optimize()

		# Check optimization status
		if m.status == GRB.Status.OPTIMAL:
			print("Assignment Status: OPTIMAL")
			print("Final Assignment Result:", m.objVal)
		else:
			print("Assignment Status: No Solution! ")
			return [[], -1, -1, []]

		# Extract the result of the T matrix
		T = [[int(lpvars[i][j].x) for j in range(col)] for i in range(row)]

		# Record the assignment pairs of the T matrix
		assignment_pairs = [(i, j) for i in range(row) for j in range(col) if T[i][j] == 1]
		assignment_agents = [i for i in range(row) for j in range(col) if T[i][j] != 0]
		assignment_pairs = sorted(assignment_pairs, key=lambda x: x[1])

		return [T, m.status, m.objVal, assignment_agents, assignment_pairs]



# generate a random Q matrix
def genRandQMat(agentNum, roleNum):
	resMat = np.zeros((agentNum, roleNum))
	for i in range(agentNum):
		for j in range(roleNum):
			resMat[i][j] = round(np.random.random(), 2)
	return resMat

# Perform dimensionality reduction on the relational matrix
# 对AC关系矩阵进行变换表示
def dimensionalityReduction(relationMat, agent_number, role_number):
	resMat = []
	for i in range(agent_number):
		for i1 in range(i+1, agent_number):
			for j in range(role_number):
				for j1 in range(role_number):
					# if i != i1:
					resMat.append([i, j, i1, j1, relationMat[i][i1]])
	return resMat


# Generate the Ac Matrix
def gen_A_c_matrix_with_conflict_times(rows, conflict_times):
	matrix = [[0] * rows for _ in range(rows)]
	recorded_idxes = []
	while conflict_times > 0:
		a, b = random.sample(range(rows), 2)
		# a = random.randint(0, rows-1)
		# b = random.choice([x for x in range(rows) if x != a])
		if a > b: a, b = b, a
		# print(a, b)
		if (a, b) in recorded_idxes:
			continue
		else:
			matrix[a][b] = 1
			matrix[b][a] = 1
			recorded_idxes.append((a, b))
			conflict_times -= 1
	return matrix


def count_relations(A, B):
	A = sorted(A)
	count = 0
	conflict_relations = []
	for i in range(len(A)):
		for j in range(i+1, len(A)):
			if B[A[i]][A[j]] == 1:
				count += 1
				conflict_relations.append((A[i], A[j]))
	return count, conflict_relations


def modify_array_randomly(row_number, t):
	"""
	Randomly modify the given 2D array 'A' based on the specified criteria.

	Parameters:
	- A (list[list[int]]): 2D input array.
	- t (int): Number of times to modify values in A.

	Returns:
	- list[list[int]]: Modified 2D array.
	"""
	A = [[0] * row_number for _ in range(row_number)]
	B = [row.copy() for row in A]  # Create a deep copy of the input array.
	n = len(A)
	if n == 0 or len(A[0]) == 0:
		return B
	possible_changes = [(i, j) for i in range(n) for j in range(n) if i != j and B[i][j] == 0 and B[j][i] == 0]
	for _ in range(t):
		if not possible_changes:
			break
		i, j = random.choice(possible_changes)
		B[i][j] = 1
		B[j][i] = 1
		possible_changes.remove((i, j))
		possible_changes.remove((j, i))
	return B

def count_zero_rows(matrix):
	"""
	Count the number of rows in a 2D array where all elements are 0.

	Parameters:
	- matrix (list[list[int]]): 2D input array.

	Returns:
	- int: Number of rows where all elements are 0.
	"""
	return sum(1 for row in matrix if all(element == 0 for element in row))

def count_one_rows(matrix):
	"""
	Count the number of rows in a 2D array where all elements are 0.

	Parameters:
	- matrix (list[list[int]]): 2D input array.

	Returns:
	- int: Number of rows where all elements are 0.
	"""
	return sum(1 for row in matrix if all(element == 1 for element in row))

def max_non_conflict_subset(Ac):
	n = len(Ac)

	# Check if two nodes are non-conflicting
	def is_valid(node, solution):
		for sol in solution:
			if Ac[node][sol] == 1:
				return False
		return True

	# Recursive function to find the max non-conflicting subset
	def backtrack(node, current_solution):
		if node == n:
			return list(current_solution)
		else:
			# Include this node
			with_node = []
			if is_valid(node, current_solution):
				with_node = backtrack(node + 1, current_solution + [node])

			# Exclude this node
			without_node = backtrack(node + 1, current_solution)

			# Return the longer solution
			return with_node if len(with_node) > len(without_node) else without_node

	# Start the recursive search from the first node
	return backtrack(0, [])


def generate_integer_list(m, n, ratio=0.8):
	target_sum = int(ratio * m)

	# Step 1: Generate initial random values
	lst = [random.randint(1, int(m * ratio)) for _ in range(n)]

	# Step 2: Calculate the difference with target sum
	current_sum = sum(lst)
	diff = current_sum - target_sum

	# Step 3: Adjust the values to get as close as possible to the target sum
	while diff != 0:
		if diff > 0:
			idx = random.randint(0, n-1)
			decrement = random.randint(1, diff)
			if lst[idx] > decrement:  # Ensure we don't go below 1
				lst[idx] -= decrement
				diff -= decrement
		else:
			idx = random.randint(0, n-1)
			increment = random.randint(1, -diff)
			if lst[idx] + increment <= int(m * ratio):  # Ensure we don't exceed m*ratio
				lst[idx] += increment
				diff += increment

	return lst

def generate_integer_list_verion_2(m, n, ratio=0.8):
	target_sum = int(ratio * m)
	if target_sum < n:
		target_sum = n

	# Step 1: Generate initial random values
	lst = [random.randint(1, int(m * ratio)) for _ in range(n)]

	# Step 2: Calculate the difference with target sum
	current_sum = sum(lst)
	diff = current_sum - target_sum

	# Step 3: Adjust the values to get as close as possible to the target sum
	while diff != 0:
		idx = random.randint(0, n-1)
		if diff > 0:
			decrement = random.randint(1, diff)
			if lst[idx] - decrement >= 1:  # Ensure we don't go below 1
				lst[idx] -= decrement
				diff -= decrement
		else:
			increment = random.randint(1, -diff)
			if lst[idx] + increment <= int(m * ratio):  # Ensure we don't exceed m*ratio
				lst[idx] += increment
				diff += increment

	return lst

# 记录实验数据 (txt)
def recordResult(fileName, records):
	with open(fileName,'a') as f:
		for record in records:
			if isinstance(record, np.ndarray):
				f.write(str(list(record)))
			else:
				f.write(str(record))
			f.write('\n')
		f.write("\n\n")


'''
 * This program is to conduct real-world scenario to verify the GRAMAC model and GRACAG model.

 * Created by Qian Jiang, September 14, 2023.
'''

# 记录实验数据 (txt)
def recordResult(fileName, records):
	with open(fileName,'a') as f:
		for key, value_list in records.items():
			# Convert numpy array to list if needed
			if any(isinstance(i, np.ndarray) for i in value_list):
				value_list = [list(i) if isinstance(i, np.ndarray) else i for i in value_list]
			f.write(f"{key}: {value_list}\n")
		f.write("\n\n")


'''
* This program is to achieve the GRA_NSGA_II Algorithm in the paper named "Iterative Role Negotiation via the Bi-level GRA++ with Decision Tolerance".

* The Python Code for the GRA-NSGA-II algorithm is created by Qian Jiang on September 14, 2023.

* Please cite the paper as the following formation: Q. Jiang, D. Liu, H. Zhu, B. Huang, N. Wu and Y. Qiao, "Group Role Assignment With Minimized Agent Conflicts," IEEE Trans. Syst. Man, Cybern. Syst., early access, Dec. 19, 2024, doi: 10.1109/TSMC.2024.3510588.
'''

def main_mba():
	file_name = 'with_conflict_GRA.txt'
	agent_number_list = [40, 80, 120, 160]
	final_record_GRAMAC_time = []
	final_record_GRAMAC_performance = []
	final_record_GRAMAC_conflict_number = []
	final_record_GRAMAC_maximum_time = []
	final_record_GRAMAC_minimum_time = []
	final_record_L_ratio = []
	for agent_number in agent_number_list:
		record_GRAMAC_time = []
		record_GRAMAC_performance = []
		record_GRAMAC_conflict_number = []
		conflict_times = agent_number
		cycling_times = 100
		for _ in range(cycling_times):
			role_number = random.randint(3, 6)
			current_ratio = random.uniform(0.8, 1)
			print("current agent number: ", agent_number)
			L = generate_integer_list_verion_2(agent_number, role_number, current_ratio)
			print(L)
			print(sum(L))
			Q_Matrix = genRandQMat(agent_number, role_number)
			A_c = gen_A_c_matrix_with_conflict_times(agent_number, conflict_times)
			max_Q_value = Q_Matrix.max()
			Q_bar = max_Q_value - np.array(Q_Matrix)
			dimension_relationMat = dimensionalityReduction(A_c, agent_number, role_number)
			time_begin_GRA = time.time()
			performance_GRA, assignment_agents_GRA, T = Assignment.GRA(Q_Matrix, L)
			time_end_GRA = time.time()
			time_GRA = time_end_GRA - time_begin_GRA
			conflict_number_GRA, _ = count_relations(assignment_agents_GRA, A_c)
			begin_time_GRAMAC = time.time()
			TMatrix_necessary_condition, result, performance, assignment_pairs_GRAMAC, assignment_agents_GRAMAC = Assignment.GRAMAC_latest_gurobi(Q_bar, Q_Matrix, L, dimension_relationMat)
			end_time_GRAMAC = time.time()
			time_GRAMAC = end_time_GRAMAC - begin_time_GRAMAC
			conflict_number_GRAMAC, _ = count_relations(assignment_agents_GRAMAC, A_c)
			performance_GRAMAC = performance
			time_cost_GRAMAC = end_time_GRAMAC-begin_time_GRAMAC
			print("GRAMAC time cost: ", time_cost_GRAMAC)
			record_GRAMAC_performance.append(performance_GRAMAC)
			record_GRAMAC_time.append(time_GRAMAC)
			record_GRAMAC_conflict_number.append(conflict_number_GRAMAC)


	print("---------overall-solution-time----------")
	print(final_record_GRAMAC_time)
	print("")

	print("---------maximum-solution-time----------")
	print("GRAMAC Maximum Time List")
	print(final_record_GRAMAC_maximum_time)
	print("")

	print("---------minimum-solution-time----------")
	print("GRAMAC Minimum Time List")
	print(final_record_GRAMAC_minimum_time)
	print("")

	print("---------overall-group-performance----------")
	print("GRAMAC Performance List")
	print(final_record_GRAMAC_performance)
	print("")

	print("---------overall-conflict-numbers----------")
	print("GRAMAC Conflict Number List")
	print(final_record_GRAMAC_conflict_number)
	print("")


	print("---------overall_ratio_L----------")
	print("overall_ratio_L")
	print(final_record_L_ratio)
	print("")



	final_records = {
		"overall_solution_time_GRAMAC": final_record_GRAMAC_time,
		"maximum_solution_time_GRAMAC": final_record_GRAMAC_maximum_time,
		"minimum_solution_time_GRAMAC": final_record_GRAMAC_minimum_time,
		"overall_group_performance_GRAMAC": final_record_GRAMAC_performance,
		"overall_conflict_numbers_GRAMAC": final_record_GRAMAC_conflict_number,
		"overall_ratio_L": final_record_L_ratio,
	}

	# Record all the results into the file
	recordResult(file_name, final_records)


if __name__ == '__main__':
	main_mba()