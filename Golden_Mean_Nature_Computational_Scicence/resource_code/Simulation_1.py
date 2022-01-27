#!/usr/bin/python3
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import var
from numpy.core.numerictypes import find_common_type
from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from math import cos, sin, acos, asin, pi, atan, fabs, ceil, floor, radians, sqrt
from collections import defaultdict
import pulp as pl
import cplex
from cplex.exceptions import CplexError
import random
import time
from matplotlib import colors, cm
from math import ceil
import csv
import copy as cp
import operator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import imageio # 生成Gif图
from sympy import *

# from The_Golden_Mean_large_scale import Q_Matrix

class Role:
	Roles_Amount = 0
	'''
	E-CARGO模型中的Role
	：id 角色标识符:
	:capacity 代理容量 L[j]:
	:priority 优先级:
	'''
	def __init__(self, role_id, role_capacity = 50, role_priority = 1):
		Role.Roles_Amount += 1
		self._id = role_id
		self._capacity = role_capacity
		self._priority = role_priority

	@property
	def id(self):
		return self._id

	@id.setter
	def id(self, val):
		self._id = val # 更新角色的标识符

	@property
	def capacity(self):
		return self._capacity # 获取Role j的L[j]值

	@capacity.setter
	def capacity(self, val):
		self._capacity = val

	@property
	def priority(self):
		return self._priority

	@priority.setter
	def priority(self, val):
		self._priority = val

class Agent:
	"""
	E-CARGO模型中的Agent
	:id 代理标识符:
	:hierachical_level 代理的评级:
	:qualificated_value_list Q[i,j]的值:
	:characteristic_type 代理的类型 characteristic_type = 0 -> 佛系， = 1 -> 自卑型, = 2 -> 越挫越勇型:
	"""
	Agents_amount = 0
	Agents_qualified_val = 0
	def __init__(self, agent_id, agent_characteristic_type, agent_hierachical_level=0):
		Agent.Agents_amount += 1
		self._id = agent_id
		self._hierachical_level = agent_hierachical_level
		# self._qualified_value = 0.40
		epsilon = 1e-3
		self._qualified_value = round(random.uniform(0+epsilon, 1-epsilon), 2)
		Agent.Agents_qualified_val += self._qualified_value
		self._characteristic_type = agent_characteristic_type

	@property
	def id(self):
		return self._id

	@id.setter
	def id(self, val):
		self._id = val # 更新代理的标识符

	@property
	def hierachical_level(self):
		return self._hierachical_level # 新代理的评级

	@hierachical_level.setter
	def hierachical_level(self, val):
		self._hierachical_level = val

	@property
	def qualified_value(self):
		return self._qualified_value

	@qualified_value.setter
	def qualified_value(self, val):
		self._qualified_value = val

	@property
	def characteristic_type(self):
		return self._characteristic_type # 代理的性格类型

	@characteristic_type.setter
	def characteristic_type(self, val):
		self._characteristic_type = val




# 利用Cplex求解问题
class Assignment:
	@classmethod
	def Cplex_GRA(cls, Q, RQ):
		"""
		: RQ 表示每个班级的平均成绩:
		"""
		row = len(Q)
		col = len(Q[0])
		student_num = row//col
		left_over_value = row % col
		L = [student_num]*(col)
		for j in range(left_over_value):
			L[j] += 1
		La = [1]*row
		vars_name = [["T"+str(i)+"_"+str(j) for j in range(col)] for i in range(row)]
		orig_vars_name = cp.deepcopy(vars_name)
		RQ_total = L * np.array(RQ)

		try:
			# 定义模型
			prob = cplex.Cplex()
			prob.objective.set_sense(prob.objective.sense.maximize)
			prob.set_log_stream(None)
			prob.set_error_stream(None)
			prob.set_warning_stream(None)
			prob.set_results_stream(None)
			orig_Q = cp.deepcopy(Q)
			Q = [i for item in Q for i in item]
			lb = [0]*(col*row)
			ub = [1]*(col*row)
			# vars_upbounds = [i for item in vars_upbounds for i in item]
			types_name = [prob.variables.type.integer]*(col*row)
			# types_name = [i for item in types_name for i in item]
			vars_name = [i for item in vars_name for i in item]
			prob.variables.add(obj=Q, lb=lb, ub=ub, types=types_name, names=vars_name)
			sense_role_constraint = "E"
			# 定义Role的约束
			role_constraints_expr = []
			role_constraints_name = []
			for j in range(0, col):
				tmp_expr_variable = [orig_vars_name[i][j] for i in range(0, row)]
				role_constraints_expr.append(cplex.SparsePair(ind=tmp_expr_variable, val=[1]*row))
				role_constraints_name.append("rc"+str(j))

			# 添加Role的约束
			prob.linear_constraints.add(lin_expr=role_constraints_expr, senses=sense_role_constraint*col, rhs=L, names=role_constraints_name)

			# 添加班级平均成绩的约束
			role_score_constraints_expr = []
			role_score_constraints_name = []
			for j in range(0, col):
				tmp_expr_variable = [orig_vars_name[i][j] for i in range(0, row)]
				Q_val = [orig_Q[i][j] for i in range(0, row)]
				role_score_constraints_expr.append(cplex.SparsePair(ind=tmp_expr_variable, val=Q_val))
				role_score_constraints_name.append("rsc"+str(j))
			prob.linear_constraints.add(lin_expr=role_score_constraints_expr, senses=sense_role_constraint*col, rhs=RQ_total, names=role_score_constraints_name)


			# 定义Agent的约束
			agent_constraints_expr = []
			agent_constraints_name = []
			for i in range(0, row):
				tmp_expr_variable = [orig_vars_name[i][j] for j in range(0, col)]
				agent_constraints_expr.append(cplex.SparsePair(ind = tmp_expr_variable, val=[1]*col))
				agent_constraints_name.append("ac"+str(i))

			# 添加Agent的约束
			prob.linear_constraints.add(lin_expr=agent_constraints_expr, senses="L"*row, rhs=La, names=agent_constraints_name)


			# 算法求解
			prob.solve()
			# 输出结果
			print("Soluiton status = ", prob.solution.status[prob.solution.get_status()])
			print("Solution value = ", prob.solution.get_objective_value())
			T = prob.solution.get_values()
			T = np.array(T)
			T = (T).reshape(row, col)
			# print("Obtained T matrix is: ", T)
			return [T, prob.solution.status[prob.solution.get_status()], prob.solution.get_objective_value(), L]

		except CplexError as exc:
			print(exc)
			return(None, None, -1, None)


	@classmethod
	def PuLP_GRA(cls, Q):
		"""
		: RQ 表示每个班级的平均成绩:
		"""
		row = len(Q)
		col = len(Q[0])
		student_num = row//col
		left_over_value = row % col
		L = [student_num]*(col)
		for j in range(left_over_value):
			L[j] += 1
		La = [1]*row

		# build a optimal problem
		pro = pl.LpProblem('Minimum the difference of the classes', pl.LpMinimize)
		# build variables for the optimal problem
		lpvars = [[pl.LpVariable("x"+str(i)+"y"+str(j), lowBound = 0, upBound = 1, cat='Binary') for j in range(col)] for i in range(row)]

		# zi表示替换|xi-RQi|的变量
		lpvars_zi = [pl.LpVariable("z_"+str(j), lowBound = 0) for j in range(col)]

		# 目标函数
		all = pl.LpAffineExpression()
		for j in range(0, col):
			all += lpvars_zi[j]

		#设定RQ的值
		RQ_val = 0
		for j in range(0, col):
			RQ_val += sum([ lpvars[i][j]*Q[i][j] for i in range(0, row)])/L[j]
		RQ_val = RQ_val/col

		pro += all

		# 添加约束 Role
		for j in range(0,col):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j], 1) for i in range(0, row)]) , 0,"L"+str(j), L[j])

		# 添加约束 Agent
		for i in range(0,row):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j], 1) for j in range(0, col)]) , 0, "La"+str(i), La[i])

		# 添加约束 去绝对值 xi-RQi <= zi
		for j in range(0, col):
			pro += sum([ lpvars[i][j]*Q[i][j] for i in range(0, row)]) <= L[j]*RQ_val+lpvars_zi[j]

		# 添加约束 去绝对值 RQi-xi <= zi
		for j in range(0, col):
			pro += sum([ lpvars[i][j]*Q[i][j] for i in range(0, row)]) >= L[j]*RQ_val-lpvars_zi[j]



		solver = pl.CPLEX_PY(warmStart = True)
		# solver.setsolver(solver)
		solver.buildSolverModel(pro)
		# for j in range(0, col):
		# 	pro.solverModel.getVars()[j].start = RQ[j] # 设置初始值
		# for j in range(col):
		# 	lpvars_zi[j].setInitialValue(ORG[j])
		solver.solverModel.parameters.timelimit.set(5)

		# solve optimal problem
		# status = pro.solve()
		solver.callSolver(pro)
		status = solver.findSolutionValues(pro)

		print("Assignment Status: ", pl.LpStatus[status])
		print("Final Assignment Result", pl.value(pro.objective))
		# get the result of T matrix
		T = [[ lpvars[i][j].varValue for j in range(col) ] for i in range(row)]
		T = np.array(T)
		T = (T).reshape(row, col)
		# print("The ideal value is: ", str(RQ_val))
		return [T, pl.value(pro.status), pl.value(pro.objective), L]


	@classmethod
	def PuLP_GRA_orig(cls, Q, RQ):
		"""
		: RQ 表示每个班级的平均成绩:
		"""
		row = len(Q)
		col = len(Q[0])
		student_num = row//col
		left_over_value = row % col
		L = [student_num]*(col)
		for j in range(left_over_value):
			L[j] += 1
		La = [1]*row

		# build a optimal problem
		pro = pl.LpProblem('Minimum the difference of the classes', pl.LpMinimize)
		# build variables for the optimal problem
		lpvars = [[pl.LpVariable("x"+str(i)+"y"+str(j), lowBound = 0, upBound = 1, cat='Binary') for j in range(col)] for i in range(row)]

		# zi表示替换|xi-RQi|的变量
		lpvars_zi = [pl.LpVariable("z_"+str(j), lowBound = 0) for j in range(col)]

		# 目标函数
		all = pl.LpAffineExpression()
		for j in range(0, col):
			all += lpvars_zi[j]


		pro += all

		# 添加约束 Role
		for j in range(0,col):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j], 1) for i in range(0, row)]) , 0,"L"+str(j), L[j])

		# 添加约束 Agent
		for i in range(0,row):
			pro += pl.LpConstraint(pl.LpAffineExpression([ (lpvars[i][j], 1) for j in range(0, col)]) , 0, "La"+str(i), La[i])

		# 添加约束 去绝对值 xi-RQi <= zi
		for j in range(0, col):
			pro += sum([ lpvars[i][j]*Q[i][j] for i in range(0, row)]) <= L[j]*(RQ[j]+lpvars_zi[j])

		# 添加约束 去绝对值 RQi-xi <= zi
		for j in range(0, col):
			pro += sum([ lpvars[i][j]*Q[i][j] for i in range(0, row)]) >= L[j]*(RQ[j]-lpvars_zi[j])

		# 添加约束 蛇形分班的threshold


		solver = pl.CPLEX_PY(warmStart = True)
		# solver.setsolver(solver)
		solver.buildSolverModel(pro)
		# for j in range(0, col):
		# 	pro.solverModel.getVars()[j].start = RQ[j] # 设置初始值
		for j in range(col):
			lpvars_zi[j].setInitialValue(RQ[j])
		solver.solverModel.parameters.timelimit.set(5)

		# solve optimal problem
		# status = pro.solve()
		solver.callSolver(pro)
		status = solver.findSolutionValues(pro)

		print("Assignment Status: ", pl.LpStatus[status])
		print("Final Assignment Result", pl.value(pro.objective))

		# get the result of T matrix
		T = [[ lpvars[i][j].varValue for j in range(col) ] for i in range(row)]
		T = np.array(T)
		T = (T).reshape(row, col)
		return [T, pl.value(pro.status), pl.value(pro.objective), L]


def haversine(task_loc, app_member_loc):
	"""
	210719 Qian Jiang
	利用两点的经纬度信息获取两点的欧式距离

	Inputs ---------------
	task_loc: 2-D list, [latitude, longitude]
	app_member_loc: 2-D list, [latitude, longitude]

	Output:
	distance: 两点之间的欧式距离, 单位为km

	test:
	task_loc = np.array([22.56614225, 113.9808368])
	member_loc = np.array([22.947097, 113.679983])
	return haversine(task_loc, member_loc)
	"""
	lat1, lon1 = task_loc
	lat2, lon2 = app_member_loc
	# 将十进制度数转化为弧度
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	# haversine公式
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a))
	earth_radius = 6371 # 地球平均半径，单位为公里
	return c * earth_radius



def genQMatrix(agents, roles):
	"""
	210729 Qian Jiang
	生成Q矩阵
	Inputs ---------------
	agents: 1-D list, [agent_1, agent_2]
	roles: 1-D list, [role_1, role_2]

	Output:
	QMatrix: 2-D list, 行（Agent）以及列（Role）
	"""
	row = len(agents)
	col = len(roles)
	QMatrix = np.zeros((row, col))
	for i in range(row):
		for j in range(col):
			QMatrix[i][j] = agents[i].qualified_value
	QMatrix = np.array(QMatrix)
	# orig_QMatrix = cp.deepcopy(QMatrix)
	# return (QMatrix-QMatrix.min())/(QMatrix.max()-QMatrix.min()), orig_QMatrix
	return QMatrix





def genMemberNewPosition(orig_app_member_loc, max_radius=0.34):
	"""
	210816 Qian Jiang
	会员的位置变化在方圆4公里的位置变化
	Inputs ---------------
	orig_app_member_loc: 2-D list, [latitude, longitude]
	max_radius: Int, 表示用户在半个小时内的最大行走距离

	Output:
	new_app_member_loc: 2-D list, 方圆3.8公里内(3.8/11的经纬度变化)的新位置
	"""
	lat, lon = orig_app_member_loc
	radius = random.uniform(0.0, max_radius)
	theta = random.random() * 2 * pi
	new_lat = lat + sin(theta) * radius
	new_lon = lon + cos(theta) * radius
	new_app_member_loc = [new_lat, new_lon]
	return new_app_member_loc


def saveLocationImage(agent_loc_list, role_loc_list, loop_times):
	# 保存当前的任务位置信息以及App member的位置信息
	# 绘制代理的位置信息
	plt.scatter(agent_loc_list[:, 1], agent_loc_list[:, 0], marker='o', s=0.5, c='k', label="App Member")
	plt.scatter(role_loc_list[:, 1], role_loc_list[:, 0], marker='o', s=0.1, c='b', label="Task")
	plt.legend(fontsize='medium')
	plt.savefig(r'./loop_%d' % (loop_times), dpi=300)
	plt.cla()




def generateGif(prefix, loop_times):
	# 生成Gif图片，不同时刻的任务以及会员的分布
	img_paths = []
	for i in range(1, loop_times+1):
		img_paths.append(prefix+'_'+str(i)+'.png')
	gif_image = []
	for path in img_paths:
		gif_image.append(imageio.imread(path))
	imageio.mimsave("res.gif", gif_image, fps=0.5)

def getKKTSolution(average_val):
	x1 = Symbol("x1")
	x2 = Symbol("x2")
	x3 = Symbol("x3")
	x4 = Symbol("x4")
	a = Symbol("a")
	b1 = Symbol("b1")
	b2 = Symbol("b2")
	b3 = Symbol("b3")
	b4 = Symbol("b4")
	f = x1**2 + x2**2 + x3**2 + x4**2 + a*(x1 + x2 + x3 + x4 - average_val*4) + b1*(-x1) + b2*(-x2) + b3*(-x3) + b4*(-x4)
	fx1 = diff(f,x1)
	fx2 = diff(f,x2)
	fx3 = diff(f, x3)
	fx4 = diff(f, x4)
	result = solve([fx1, fx2, fx3, fx4, b1*-x1, b2*-x2, b3*-x3, b4*-x4, x1+x2+x3+x4-average_val*4],[x1,x2, x3, x4, a, b1, b2, b3, b4])

	res = []
	for item in result:
		if (np.array(item[:4]) > 0).all():
			res.append(item[:4])
	minResult = float('inf')
	minItem = []
	for item in res:
		x_1, x_2, x_3, x_4 = item
		objective_val = x_1**2 + x_2**2 + x_3**2 + x_4**2
		if objective_val < minResult:
			minResult = objective_val
			minItem = item
	return minItem


def serpentineQueue(agent_list, role_list):
	row = len(agent_list)
	col = len(role_list)
	student_num = row//col
	left_over_num = row % col
	T_Matrix = np.zeros((row, col))
	serpentine_queue = []
	for i in range(row):
		serpentine_queue.append((agent_list[i].id, agent_list[i].qualified_value))
	serpentine_queue = sorted(serpentine_queue, key = lambda x: x[1], reverse = True)
	tmp_queue = []
	for (id, val) in serpentine_queue:
		tmp_queue.append(id)
	serpentine_queue = tmp_queue
	left_over_queue = cp.deepcopy(serpentine_queue[student_num*col:])
	serpentine_queue = serpentine_queue[:student_num*col]
	serpentine_queue = np.array(serpentine_queue).reshape(student_num, col)
	# serpentine_queue = list(serpentine_queue)
	for i in range(student_num):
		if i % 2 == 1:
			serpentine_queue[i] = serpentine_queue[i][::-1]
	for j in range(col):
		for i in range(len(serpentine_queue)):
			T_Matrix[serpentine_queue[i][j]][j] = 1
	if left_over_num != 0:
		# 对最后一行特殊处理
		if student_num % 2 == 0:
			i = 0
			for j in range(0, left_over_num):
				T_Matrix[left_over_queue[i]][j] = 1
				i += 1
		else:
			# left_over_queue = left_over_queue[::-1]
			i = 0
			for j in range(col-1, col-1-left_over_num, -1):
				T_Matrix[left_over_queue[i]][j] = 1
				i += 1
	return T_Matrix



if __name__ == '__main__':
	# 初始化参数
	agent_number = 200 # 人的数量
	role_number = 4 # 班级的数量
	cycling_times = 100 # 循环次数
	# 记录结果
	RQ_res_list = []
	cplex_result_list = []
	cplex_result_orig_list = []
	serpentine_result_list = []
	for current_time in range(cycling_times):
		# 生成随机的角色
		role_list = []
		for j in range(role_number):
			role_id = j
			role_list.append(Role(role_id))
		print(role_list)
		agent_list = []
		for i in range(agent_number):
			agent_id = i
			agent_type = random.randint(0, 2)
			agent_list.append(Agent(agent_id, agent_type))
		Q = genQMatrix(agent_list, role_list)
		Q = np.array(Q)
		serpentine_T_Matrix = serpentineQueue(agent_list, role_list)
		average_val = Q.sum(axis=0)/role_number
		RQ = [average_val]*role_number
		RQ_res = Q.sum(axis=0)/agent_number
		# CPlex求解
		T_Matrix, result, performance, L = Assignment.PuLP_GRA(Q)
		L = np.array(L)
		T_Matrix_orig, result_orig, performance_orig, L_orig = Assignment.PuLP_GRA_orig(Q, RQ_res)
		cplex_result_orig = (Q*T_Matrix_orig).sum(axis = 0)/L_orig
		cplex_result = (Q*T_Matrix).sum(axis = 0)/L
		serpentine_result = (Q*serpentine_T_Matrix).sum(axis = 0)/serpentine_T_Matrix.sum(axis = 0)
		print(RQ_res)
		print(cplex_result)
		print(cplex_result_orig)
		print(serpentine_result)
		RQ_res_list.append(RQ_res)
		cplex_result_list.append(cplex_result)
		cplex_result_orig_list.append(cplex_result_orig)
		serpentine_result_list.append(serpentine_result)
		print(T_Matrix_orig.sum(axis=0))
		print(T_Matrix.sum(axis=0))
	RQ_res_list = np.array(RQ_res_list)
	cplex_result_list = np.array(cplex_result_list)
	cplex_result_orig_list = np.array(cplex_result_orig_list)
	serpentine_result_list = np.array(serpentine_result_list)
	print("The output result of Simluation 1:")
	print(RQ_res_list.sum(axis = 0)/cycling_times)
	print(cplex_result_list.sum(axis = 0)/cycling_times)
	print(cplex_result_orig_list.sum(axis = 0)/cycling_times)
	print(serpentine_result_list.sum(axis = 0)/cycling_times)
	print("Testing ------")
	print(sum(RQ_res_list.sum(axis = 0)/cycling_times))
	print(sum(cplex_result_list.sum(axis = 0)/cycling_times))
	print(sum(cplex_result_orig_list.sum(axis = 0)/cycling_times))
	print(sum(serpentine_result_list.sum(axis = 0)/cycling_times))
	optimal_value = []
	# print(list(cplex_result))
	# print(list(serpentine_result))
	# print(pulp_cplex_result)
	# print(sum(RQ) - sum(cplex_result))
	# print(sum(RQ) - sum(serpentine_result))
	# print(sum(RQ) - sum(pulp_cplex_result))
	# if (T_Matrix == T_Matrix_pulp).all():
		# print("T_Matrix == T_Matrix_pulp")
	# else:
		# print("No!")
	# if (T_Matrix == serpentine_T_Matrix).all():
	# 	print("T-Matrix == serpentine_T_Matrix")
	# else:
	# 	print("No!")
	# if (serpentine_T_Matrix == T_Matrix_pulp).all():
	# 	print("serpentine_T_Matrix == T_Matrix_pulp")
	# else:
	# 	print("No!")
