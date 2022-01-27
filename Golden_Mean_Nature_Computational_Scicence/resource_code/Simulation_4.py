#!/usr/bin/python3
from os import write
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import alltrue, var
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
import imageio
from pulp.apis.core import initialize # 生成Gif图
from sympy import *
from sympy.core.expr import unchanged
import xlrd
import xlwt
from xlutils.copy import copy
import openpyxl


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
	>>> Qian Jiang 211205
	>>> E-CARGO模型中的Agent
	`id`: 代理标识符, int
	`hierachical_level`: int, 代理的评级
	`qualificated_value_list`: list, Q[i,j]的值
	`characteristic_type`: int, 代理的类型 characteristic_type = 0 -> 佛系， = 1 -> 自卑型, = 2 -> 越挫越勇型:
	"""
	Agents_amount = 0
	Agents_qualified_val = 0
	def __init__(self, agent_id, agent_characteristic_type, agent_hierachical_level=0):
		Agent.Agents_amount += 1
		self._id = agent_id
		self._hierachical_level = agent_hierachical_level
		# self._qualified_value = 0.40
		self._qualified_value = round(random.uniform(0, 1), 2)
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
	def PuLP_GRA(cls, Q, RQ):
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


# 生成情绪牵制率 alpha, alpha表示比自己优秀的人的数量比例
def generateAlphaList(qualified_vector, solution_dics):
	'''
	>>> Qian Jiang 20211207
	>>> --- input
	`qualified_vector`: list, 表示每个学生所对应的Q[i, j](t-1)的值
	`solution_dics`： dictionary, 表示分班的字典，如solution_dic[0] = 1 表示agent 0被分到role 1.
	>>> --- output
	`alpha_list`: list, 表示学习率列表
	'''
	alpha_list = []
	for i in range(len(qualified_vector)):
		class_i = solution_dics[i]
		class_i_idxs = [item[0] for item in solution_dics.items() if item[1] == class_i]
		class_number = len(class_i_idxs) # 班级数量
		qualified_vector = np.array(qualified_vector)
		class_scores = qualified_vector[class_i_idxs]
		alpha = sum(class_scores > qualified_vector[i])/class_number
		alpha_list.append(alpha)
	return alpha_list

def generateRandomAlphaList(qualified_vector):
	alpha_list = []
	agent_number = len(qualified_vector)
	epsilon = 1e-2
	for i in range(agent_number):
		alpha_i = round(random.uniform(0+epsilon, 1/2-epsilon), 2)
		alpha_list.append(alpha_i)
	return alpha_list

# 生成学习率beta向量
def generateBetaList(qualified_vector, student_number, Base = 1/5):
	'''
	>>> Qian Jiang 20211201
	>>> --- input
	`qualified_vector`: list, 表示每个学生所对应的Q[i, j](t-1)的值
	`period_time`: int, 表示班级/系存在周期
	`student_number`: int, 表示系的总人数
	`Base`: float, 表示指数函数的底数，Base in (0, 1)
	>>> --- output
	`beta_list`: list, 表示qualified_vector中的学生对应的学习率，由小到大
	'''
	beta_list = []
	end_x = 1 # 最小的取值为1，随机取某一段作为beta的变化率
	interval = 1/student_number
	x_list = np.arange(interval, end_x+interval, interval)
	for item in x_list:
		beta_list.append(Base**item)
	# 利用qualified_vector对beta_list进行排序
	beta_list = np.array(beta_list)
	qualified_vector = np.array(qualified_vector)
	idxs = qualified_vector.argsort()[::-1]
	sorted_beta_list = np.zeros(student_number)
	for i in range(len(idxs)):
		sorted_beta_list[idxs[i]] = beta_list[i]
	return sorted_beta_list


# 根据Agent的个性生成持续变化的Q值
def generate_DGRA_Q_value(Q_current, alpha_list, beta_list, agent_list, t):
	'''
	>>> Qian Jiang 20211205
	>>> --- input
	`Q_val`: int, 表示Agent当前的Q值.
	`alpha_list`: list, 情绪牵制率alpha列表.
	`beta_list`: list, 学习率beta列表.
	`agent_list`: list, 代理列表
	`t`: float, 当前时间.
	>>> --- output
	`Q_latest_val`: int, 表示每个学生对应的学习率.
	'''
	row = len(Q_current)
	col = len(Q_current[0])
	Q_latest = np.zeros((row, col))
	for i in range(row):
		for j in range(col):
			alpha = alpha_list[i]
			beta = beta_list[i]
			Q_val = Q_current[i][j]
			agent_type = agent_list[i].characteristic_type
			Q_latest_val = None
			if agent_type == 0: # 佛系人格
				epsilon = 1e-4
				omega = 10
				Q_latest_val = epsilon*np.sin(omega*t)+beta*(-np.e**(-t)+1)+Q_val
			elif agent_type == 1: # 自卑型人格
				# alpha*np.sin(2*t-np.pi/2)-2*alpha*np.cos(t-np.pi)+alpha -> 情绪牵制部分
				tmp_alpha = alpha/4 # 保证Q_latest_val > 0
				Q_latest_val = tmp_alpha*np.sin(2*t-np.pi/2) - 2*tmp_alpha*np.cos(t-np.pi) - tmp_alpha + beta*(-np.e**(-t)+1) + Q_val
			elif agent_type == 2: # 越挫越勇型人格
				# e.g., 1/2*sin(x-pi/2)+1/2
				tmp_alpha = alpha/2
				Q_latest_val = tmp_alpha*np.sin(t-np.pi/2) + tmp_alpha + beta*(-np.e**(-t)+1) + Q_val
			else:
				assert("Error input of the agent type!")
			Q_latest[i][j] = Q_latest_val
	# 更新agent的Q值
	for i in range(len(agent_list)):
		agent_list[i].qualified_value = Q_latest[i][0]
	return np.array(Q_latest)

# 根据Agent的个性生成连续累计的Q值，即对时间的积分
def generateContinuous_Q(start_time, end_time, expression_part):
	'''
	>>> Qian Jiang 20211210
	>>> --- input
	`start_time`: float, 开始时间.
	`end_time`: float, 结束时间.
	`expression_part`: tuple, 表示agent的性格以及函数的各个组成部分, agent_characteristics_type = 0, y = 2*sinx+ 3x + 3 -> (0, 2, 3, 3).
	>>> --- output
	`Q_continuous_val`: float, 表示某个学生在某个时段的Q值.
	'''
	agent_type, alpha, beta, Q_val = expression_part
	epsilon = 1e-2
	omega = 10
	Q_latest_val = None
	# x = symbols('x')
	if end_time > start_time:
		if agent_type == 0: # 佛系人格
			# Q_latest_val = epsilon*np.sin(omega*t)+beta*(-np.e**(-t)+1)+Q_val
			Q_latest_val =  (-epsilon*np.cos(omega*end_time)/omega+beta*np.e**(-end_time)+(beta+Q_val)*end_time) - (-epsilon*np.cos(omega*start_time)/omega+beta*np.e**(-start_time)+(beta+Q_val)*start_time)
		elif agent_type == 1: # 自卑型人格
			# alpha*np.sin(2*t-np.pi/2)-2*alpha*np.cos(t-np.pi)+alpha -> 情绪牵制部分
			tmp_alpha = alpha/4 # 保证Q_latest_val > 0
			Q_latest_val = (-tmp_alpha*np.cos(2*end_time-np.pi/2)/2 - 2*tmp_alpha*np.sin(end_time-np.pi) - tmp_alpha*end_time + beta*np.e**(-end_time)+(beta+Q_val)*end_time) - (-tmp_alpha*np.cos(2*start_time-np.pi/2)/2 - 2*tmp_alpha*np.sin(start_time-np.pi) - tmp_alpha*start_time + beta*np.e**(-start_time)+(beta+Q_val)*start_time)
		elif agent_type == 2: # 越挫越勇型人格
			# e.g., 1/2*sin(x-pi/2)+1/2
			tmp_alpha = alpha/2
			# Q_latest_val = tmp_alpha*np.sin(t-np.pi/2) + tmp_alpha + beta*(-np.e**(-t)
			# +1) + Q_val
			Q_latest_val = (-tmp_alpha*np.cos(end_time-np.pi/2) + tmp_alpha*end_time + beta*np.e**(-end_time)+(beta+Q_val)*end_time) - (-tmp_alpha*np.cos(start_time-np.pi/2) + tmp_alpha*start_time + beta*np.e**(-start_time)+(beta+Q_val)*start_time)
		else:
			assert("Error input of the agent type!")
	else:
		Q_latest_val = Q_val
	return Q_latest_val

# 根据Agent的个性生成连续累计的Q值，即对时间的积分
def generateContinuous_Q_random_alpha(start_time, end_time, expression_part):
	'''
	>>> Qian Jiang 20211210
	>>> --- input
	`start_time`: float, 开始时间.
	`end_time`: float, 结束时间.
	`expression_part`: tuple, 表示agent的性格以及函数的各个组成部分, agent_characteristics_type = 0, y = 2*sinx+ 3x + 3 -> (0, 2, 3, 3).
	>>> --- output
	`Q_continuous_val`: float, 表示某个学生在某个时段的Q值.
	'''
	agent_type, alpha, _, Q_val = expression_part
	epsilon = 1e-2
	omega = 10
	Q_latest_val = None
	# x = symbols('x')
	if end_time > start_time:
		if agent_type == 0: # 佛系人格
			# Q_latest_val = epsilon*np.sin(omega*t)+beta*(-np.e**(-t)+1)+Q_val
			Q_latest_val =  Q_val*end_time - Q_val*start_time
		elif agent_type == 1: # 自卑型人格
			# alpha*np.sin(4*t-np.pi/2) -> 情绪牵制部分
			tmp_alpha = -alpha # 保证Q_latest_val > 0
			Q_latest_val = (-tmp_alpha*np.cos(4*end_time-np.pi/2)/4 + (Q_val+tmp_alpha)*end_time) - (-tmp_alpha*np.cos(4*start_time-np.pi/2)/4 + (Q_val+tmp_alpha)*start_time)
		elif agent_type == 2: # 越挫越勇型人格
			# e.g., 1/2*sin(x-pi/2)+1/2
			tmp_alpha = alpha
			# Q_latest_val = tmp_alpha*np.sin(t-np.pi/2) + tmp_alpha + beta*(-np.e**(-t)
			# +1) + Q_val
			Q_latest_val = (-tmp_alpha*np.cos(4*end_time-np.pi/2)/4 + (Q_val+tmp_alpha)*end_time) - (-tmp_alpha*np.cos(4*start_time-np.pi/2)/4 + (Q_val+tmp_alpha)*start_time)
		else:
			assert("Error input of the agent type!")
	else:
		Q_latest_val = Q_val
	return Q_latest_val





# 生成指定比例人格的agent
def generateAgentList(agent_type_list, agent_number):
	'''
	>>> Qian Jiang 20211206
	>>> --- input
	`agent_type_list`: int, 表示当前agent的不同人格数量. e.g., agent_type_list = [60, 70, 70], 同时len(agent_type_list) == agent_number.
	>>> --- output
	`agent_list`: list, 表示agent的性格的list.
	'''
	agent_list = []
	agent_type_0, agent_type_1, agent_type_2 = agent_type_list
	type_list = [0]*agent_type_0 + [1]*agent_type_1 + [2]*agent_type_2
	random.shuffle(type_list) #打乱顺序
	for i in range(agent_number):
		agent_id = i
		agent_type = type_list[i]
		agent_list.append(Agent(agent_id, agent_type))
	return agent_list



# 获取T矩阵的可行解对
def getSolutionUnits(T_matrix):
	# solution_units = []
	solution_dic = {}
	row = len(T_matrix)
	col = len(T_matrix[0])
	epsilon = 1e-4
	for i in range(row):
		for j in range(col):
			if abs(T_matrix[i][j]-1) < epsilon:
				# solution_units.append((i, j))
				solution_dic[i] = j
	# solution_units = sorted(solution_units, key= lambda x:x[1])

	# for (key, val) in solution_units:
		# solution_dic[key] = val
	return solution_dic

# 获取可行的集合
def getSolutionSets(T_Matrix):
	solution_dics = {}
	row = len(T_Matrix)
	col = len(T_Matrix[0])
	epsilon = 1e-4
	for j in range(col):
		solution_dics[j] = set()
	for i in range(row):
		for j in range(col):
			if abs(T_Matrix[i][j]-1) < epsilon:
				solution_dics[j].add(i)
	return solution_dics


# 获取班级的前20%的成绩
def getTop20Value(Q_Matrix, T_Matrix):
	row = len(Q_Matrix)
	col = len(Q_Matrix[0])
	epsilon = 1e-4
	class_num = row//col
	Num_20percent = int(np.ceil(class_num*0.2))
	class_performance = []
	for j in range(col):
		tmp_list = []
		for i in range(row):
			if abs(T_Matrix[i][j]-1) < epsilon:
				tmp_list.append(Q_Matrix[i][j])
		tmp_list = sorted(tmp_list, reverse=True)
		class_performance.append(tmp_list)
	res = np.zeros(col)
	for j in range(col):
		res[j] = sum(class_performance[j][:Num_20percent])/Num_20percent
	res = sorted(res, reverse=True)
	return res

# 获取班级的后20%的成绩
def getLast20Value(Q_Matrix, T_Matrix):
	row = len(Q_Matrix)
	col = len(Q_Matrix[0])
	epsilon = 1e-4
	class_num = row//col
	Num_20percent = int(np.ceil(class_num*0.2))
	class_performance = []
	for j in range(col):
		tmp_list = []
		for i in range(row):
			if abs(T_Matrix[i][j]-1) < epsilon:
				tmp_list.append(Q_Matrix[i][j])
		tmp_list = sorted(tmp_list, reverse=False)
		class_performance.append(tmp_list)
	res = np.zeros(col)
	for j in range(col):
		res[j] = sum(class_performance[j][:Num_20percent])/Num_20percent
	res = sorted(res, reverse=True)
	return res

# 获取系前百分之20%的成绩
def getTop20Value_Global(qualified_vector):
	tmp_qualified_vector = sorted(qualified_vector, reverse=True)
	return sum(tmp_qualified_vector[:int(np.ceil(int(0.2*len(tmp_qualified_vector))))])

# 获取系后百分之20%的成绩：
def getLast20Value_Global(qualified_vector):
	tmp_qualified_vector = sorted(qualified_vector, reverse=False)
	return sum(tmp_qualified_vector[:int(np.ceil(int(0.2*len(tmp_qualified_vector))))])

# 找到学生成绩变化的列表
def getChangedStudents(orig_qualified_vector, qualified_vector_lastest, agent_list):
	# res = []
	# epsilon = 1e-4
	length = len(orig_qualified_vector)
	increase_list = []
	decrease_list = []
	unchanged_list = []
	for k in range(len(qualified_vector_lastest)):
		if qualified_vector_lastest[k] - orig_qualified_vector[k] < 0:
			decrease_list.append([k, agent_list[k].characteristic_type, orig_qualified_vector[k], qualified_vector_lastest[k], length-sum(orig_qualified_vector < orig_qualified_vector[k])])
		elif qualified_vector_lastest[k] - orig_qualified_vector[k] > 0:
			increase_list.append([k, agent_list[k].characteristic_type, orig_qualified_vector[k], qualified_vector_lastest[k], length-sum(orig_qualified_vector < orig_qualified_vector[k])])
		else:
			unchanged_list.append([k, agent_list[k].characteristic_type, orig_qualified_vector[k], qualified_vector_lastest[k], length-sum(orig_qualified_vector < orig_qualified_vector[k])])
	increase_list = sorted(increase_list, key= lambda x:x[-1])
	unchanged_list = sorted(unchanged_list, key= lambda x:x[-1])
	decrease_list= sorted(decrease_list, key= lambda x:x[-1])
	return increase_list, unchanged_list, decrease_list

# 记录实验数据 (excel)
def record_data(data_set, type = 0):
	path = 'dataList_compare_absolute_relative_assignment.xlsx'
	if type == 0: # 增长的agent
		sheetName = 'increased_list'
	elif type == 1: # 不变的agent
		sheetName = 'unchanged_list'
	elif type == 2: # 减少的agent
		sheetName = 'decreased_list'
	else:
		assert("Error input of the type!")
	data_set = np.array(cp.deepcopy(data_set))
	df = pd.DataFrame(data_set)
	# writer = pd.ExcelWriter(path, engine='openpyxl', mode='a')
	# df.to_excel(writer, sheet_name, index=0, header=0)
	with pd.ExcelWriter(path, mode="a", engine="openpyxl") as writer:
		df.to_excel(writer, sheet_name=sheetName, index = 0, header = 0)
	writer.save()

# 记录实验数据 (txt)
def recordResult(fileName, records):
	with open(fileName,'a') as f:
		for record in records:
			f.write(str(record))
			f.write('\n')
		# f.write("\n\n")

# 记录不同分数层级的人的变化
def record_hierarchical_scores(qualified_vector, hierarchical_interval = 0.2):
	res_list = [0]
	interval_list = [k for k in np.arange(hierarchical_interval, 1, hierarchical_interval)]
	interval_list = [round(item, 2) for item in interval_list]
	total_performance = sum(qualified_vector)
	tmp_qualified_vector = sorted(qualified_vector, reverse=True)
	for interval in interval_list:
		tmp_val = sum(tmp_qualified_vector[:int(np.ceil(int(interval*len(tmp_qualified_vector))))]) - sum(res_list)
		res_list.append(tmp_val)
	res_list.append(total_performance-sum(res_list))
	return res_list[1:]

# 调整指定人格类型的agent的Q value
def adjustingAgentValues(agent_list, adjusting_agent_type, hierarchical_level = 0):
	'''
	>>> Qian Jiang 20211213
	>>> --- input
	`agent_list`: 1-D list, 表示agent的列表
	`adjusting_agent_type`: int, 表示需要调整Q value的agent类型
	`hierarchical_level`: int, 表示需要修改的agent的层级, hierarchical_level = 0 -> Top 20%, = 1 -> Top 20-40%, = 2 Top 40%->60%, = 3 Top 60%->80%, = 4 least 80%.
	>>> --- output
	None
	'''
	adjusting_idx_list = []
	for i in range(len(agent_list)):
		if agent_list[i].characteristic_type == adjusting_agent_type:
			adjusting_idx_list.append(i)
	initial_qualified_vector = [item.qualified_value for item in agent_list]
	agent_number = len(agent_list)
	res_list = []
	hierarchical_interval = 0.2
	interval_list = [k for k in np.arange(hierarchical_interval, 1 + hierarchical_interval, hierarchical_interval)]
	interval_list = [round(item, 2) for item in interval_list]
	tmp_qualified_vector = sorted(initial_qualified_vector, reverse=True)
	for interval in interval_list:
		tmp_list = []
		for item in res_list:
			tmp_list += item
		tmp_val = cp.copy(tmp_qualified_vector[:int(np.ceil(int(interval*agent_number)))])
		for item in tmp_list:
			tmp_val.remove(item)
		res_list.append(tmp_val)
	epsilon = 1e-4
	min_val = min(res_list[hierarchical_level])
	max_val = max(res_list[hierarchical_level])
	for idx in adjusting_idx_list:
		agent_list[idx].qualified_value = round(random.uniform(min_val+epsilon, max_val-epsilon), 2)

# 记录实验数据 (txt)
def recordResult(fileName, records):
	with open(fileName,'a') as f:
		for record in records:
			f.write(str(record))
			f.write('\n')
		# f.write("\n\n")




'''
 * This program is to conduct large-scale experiments to compare the absolute fairness and the relative fairness  "The Golden Mean".

 * Created by Qian Jiang, December 12, 2021.
'''

if __name__ == '__main__':
	# 初始化参数
	agent_number = 202
	role_number = 4 # 班级的数量
	start_time = 0 # 学期开始的时间
	end_time = 4*np.pi # 学期完成时间
	time_interval = np.pi/2 # 学期间隔
	cyclingTimes = 1 # 大规模实验的循环次数
	adjusted_agent_type = 1 # 需要调整的agent的人格类型
	b_val = 0.001
	time_list = [k for k in np.arange(time_interval, end_time + time_interval, time_interval)]
	record_agent_type_list = [[192, 10, 0]]
	# record_agent_type_list = [[92, 10, 100], [112,  10, 80], [152, 10, 40], [172, 10, 20], [182, 10, 10], [172, 20, 10], [152, 40, 10], [112, 80, 10], [92, 100, 10]]
	for agent_type_list in record_agent_type_list:
		# 生成随机的角色
		role_list = []
		fileName = "simulation5.txt"
		for j in range(role_number):
			role_id = j
			role_list.append(Role(role_id))
		agent_list = []
		record = []
		# agent_type_list = [182, 10, 10]
		orig_total_performance_list = []
		latest_total_performance_list = []
		orig_top_20_percent_list = []
		orig_last_20_percent_list = []
		latest_top_20_percent_list = []
		latest_last_20_percent_list = []
		cplex_result_list = []
		orig_cplex_result_list = []
		# orig_max_agent_list = []
		# orig_last_agent_list = []
		# latest_max_agent_list = []
		# latest_last_agent_list = []
		orig_hierarchical_list = []
		latest_hierarchical_list = []
		roles_performance_list_relative_equality_allocation = [] # 记录所有的roles的overall performances
		roles_performance_list_absolute_equality_allocation = []
		modified_amount_list = []
		for time in range(cyclingTimes):
			flag = 1 # 用于第一次的适应
			roles_performance_relative_equality_allocation = []
			roles_performance_absolute_equality_allocation = []
			agent_list = generateAgentList(agent_type_list, agent_number)
			# 调整指定人格类型的agent的Q value
			# adjustingAgentValues(agent_list, adjusted_agent_type)
			Q = genQMatrix(agent_list, role_list)
			orig_qualified_vector = [item.qualified_value for item in agent_list]
			orig_average_val = sum(orig_qualified_vector)/agent_number
			RQ = [orig_average_val]*role_number
			# beta不受Q的限制，但是alpha受Q的限制
			beta_list = generateBetaList(orig_qualified_vector, agent_number, b_val)
			alpha_list = generateRandomAlphaList(orig_qualified_vector)
			orig_T_Matrix, orig_result, orig_performance, orig_L = Assignment.PuLP_GRA(Q, RQ)
			orig_L = np.array(orig_L)
			initial_roles_performance = (orig_T_Matrix*Q).sum(axis = 0)/orig_L
			roles_performance_absolute_equality_allocation.append(initial_roles_performance)
			roles_performance_relative_equality_allocation.append(initial_roles_performance)
			orig_solution_dics = getSolutionUnits(orig_T_Matrix)
			# alpha_list = generateAlphaList(orig_qualified_vector, orig_solution_dics)
			# orig_alpha_list = generateAlphaList(orig_qualified_vector, orig_solution_dics)
			orig_alpha_list = [0]*len(agent_list)
			Q = np.array(Q)
			Q_latest = np.zeros((agent_number, role_number))
			orig_Q_assignment = np.zeros((agent_number, role_number))
			for current_time in time_list:
				for i in range(len(Q_latest)):
					for j in range(len(Q_latest[0])):
						Q_latest[i][j] += generateContinuous_Q_random_alpha(current_time-time_interval, current_time, (agent_list[i].characteristic_type, alpha_list[i], beta_list[i], Q[i][j]))
						if flag == 1:
							orig_Q_assignment[i][j] += generateContinuous_Q_random_alpha(current_time-time_interval, current_time, (agent_list[i].characteristic_type, alpha_list[i], beta_list[i], Q[i][j]))
							flag = 0
						else:
							orig_Q_assignment[i][j] += generateContinuous_Q_random_alpha(current_time-time_interval, current_time, (agent_list[i].characteristic_type, orig_alpha_list[i], beta_list[i], Q[i][j]))
				# 更新agent的Q值
				qualified_vector_latest = [Q_latest[i][0] for i in range(agent_number)]
				orig_qualified_vector = [orig_Q_assignment[i][0] for i in range(agent_number)]
				for i in range(len(agent_list)): # 更新agent的qualified_value
					agent_list[i].qualified_value = qualified_vector_latest[i]
				serpentine_T_Matrix = serpentineQueue(agent_list, role_list)
				serpentine_result = (Q_latest*serpentine_T_Matrix).sum(axis = 0)/serpentine_T_Matrix.sum(axis = 0)
				average_val = sum(qualified_vector_latest)/agent_number
				RQ = [average_val]*role_number
				T_Matrix, result, performance, L = Assignment.PuLP_GRA(Q_latest, RQ)
				L = np.array(L)
				solution_dics = getSolutionUnits(serpentine_T_Matrix)
				# print(alpha_list)
				# print(orig_alpha_list)
				# print(RQ)
				cplex_result = (Q_latest*T_Matrix).sum(axis = 0)/L
				roles_performance_absolute_equality_allocation.append(cplex_result)
				orig_cplex_result = (orig_T_Matrix*orig_Q_assignment).sum(axis = 0)/orig_L
				roles_performance_relative_equality_allocation.append(orig_cplex_result)
				# print(cplex_result)
				# print(orig_cplex_result)
				# print(serpentine_result)
				T_Matrix_set = getSolutionSets(T_Matrix)
				orig_T_Matrix_set = getSolutionSets(orig_T_Matrix)
			cplex_result_list.append(cplex_result)
			orig_cplex_result_list.append(orig_cplex_result)
				# print(T_Matrix_set.values())
				# print(orig_T_Matrix_set.values())
			# print("Max Q value of the original Q_Matrix is: ", Q.max())
			roles_performance_absolute_equality_allocation = np.array(roles_performance_absolute_equality_allocation)
			roles_performance_list_absolute_equality_allocation.append(roles_performance_absolute_equality_allocation)
			roles_performance_relative_equality_allocation = np.array(roles_performance_relative_equality_allocation)
			roles_performance_list_relative_equality_allocation.append(roles_performance_relative_equality_allocation)
			orig_top_20_percent = getTop20Value_Global(orig_qualified_vector)
			orig_last_20_percent = getLast20Value_Global(orig_qualified_vector)
			print("The top 20% - one-time assignment", orig_top_20_percent)
			print("The last 20% - one-time assignment", orig_last_20_percent)
			orig_top_20_percent_list.append(orig_top_20_percent)
			orig_last_20_percent_list.append(orig_last_20_percent)
			max_orig_vector_value = max(orig_qualified_vector)
			max_orig_vector_value_index = orig_qualified_vector.index(max_orig_vector_value)
			max_orig_vector_value_characteristic_type = agent_list[max_orig_vector_value_index].characteristic_type
			min_orig_vector_value = min(orig_qualified_vector)
			min_orig_vector_value_index = orig_qualified_vector.index(min_orig_vector_value)
			min_orig_vector_value_characteristic_type = agent_list[min_orig_vector_value_index].characteristic_type
			print("Total performance - one-time assignment: ", sum(orig_qualified_vector))
			orig_total_performance_list.append(sum(orig_qualified_vector))
			print("Max agent - one-time assignment: %f, Agent %d, Characteristic_type: %d" % (max_orig_vector_value, max_orig_vector_value_index, max_orig_vector_value_characteristic_type))
			print("Initial Q value of the Agent %d is %f " % (max_orig_vector_value_index, Q[max_orig_vector_value_index][0]))
			print("Min agent - one-time assignment: %f, Agent %d, Characteristic_type: %d" % (min_orig_vector_value, min_orig_vector_value_index, min_orig_vector_value_characteristic_type))
			# print("Top 20% - one-time assignment: ")
			# latest_top_20_percent = getTop20Value(Q_latest, T_Matrix)
			# latest_last_20_percent = getLast20Value(Q_latest, T_Matrix)
			latest_top_20_percent = getTop20Value_Global(qualified_vector_latest)
			latest_last_20_percent = getLast20Value_Global(qualified_vector_latest)
			print("The top 20% - re-assignment", latest_top_20_percent)
			print("The last 20% - re-assignment", latest_last_20_percent)
			latest_top_20_percent_list.append(latest_top_20_percent)
			latest_last_20_percent_list.append(latest_last_20_percent)
			max_qualified_vector_latest = max(qualified_vector_latest)
			max_qualified_vector_latest_index = qualified_vector_latest.index(max_qualified_vector_latest)
			max_qualified_vector_latest_characteristic_type = agent_list[max_qualified_vector_latest_index].characteristic_type
			min_qualified_vector_latest = min(qualified_vector_latest)
			min_qualified_vector_latest_index = qualified_vector_latest.index(min_qualified_vector_latest)
			min_qualified_vector_latest_characteristic_type = agent_list[min_qualified_vector_latest_index].characteristic_type
			print("Total performance - re-assignment: ", sum(qualified_vector_latest))
			latest_total_performance_list.append(sum(qualified_vector_latest))
			print("Max agent - re-assignment: %f, Agent %d, Characteristic_type: %d" % (max_qualified_vector_latest, max_qualified_vector_latest_index, max_qualified_vector_latest_characteristic_type))
			print("Initial Q value of the Agent %d is %f " % (max_qualified_vector_latest_index, Q[max_qualified_vector_latest_index][0]))
			print("Min agent - re-assignment: %f, Agent %d, Characteristic_type: %d" % (min_qualified_vector_latest, min_qualified_vector_latest_index,min_qualified_vector_latest_characteristic_type))
			print("The minimum agent %d of the original assignment is: %f" % (min_orig_vector_value_index,qualified_vector_latest[min_orig_vector_value_index]))

			increase_list, unchanged_list, decrease_list = getChangedStudents(orig_qualified_vector, qualified_vector_latest, agent_list)
			modified_amount_list.append([len(increase_list), len(unchanged_list), len(decrease_list)])
			orig_hierarchical_res = record_hierarchical_scores(orig_qualified_vector)
			latest_hierachical_res = record_hierarchical_scores(qualified_vector_latest)
			orig_hierarchical_list.append(orig_hierarchical_res)
			latest_hierarchical_list.append(latest_hierachical_res)
		roles_performance_list_absolute_equality_allocation = np.array(roles_performance_list_absolute_equality_allocation)
		roles_performance_list_relative_equality_allocation = np.array(roles_performance_list_relative_equality_allocation)
		record.append("----------"+str(b_val)+"------------"+str(cyclingTimes)+"---------")
		record.append(str(agent_type_list))
		record.append("------One-time Assignment Result------")
		record.append(str(sum(orig_total_performance_list)/cyclingTimes))
		record.append(str(sum(orig_top_20_percent_list)/cyclingTimes))
		record.append(str(sum(orig_last_20_percent_list)/cyclingTimes))
		orig_hierarchical_list = np.array(orig_hierarchical_list)
		record.append(str(orig_hierarchical_list.sum(axis = 0)/cyclingTimes))
		# print(roles_performance_absolute_equality_allocation)
		# print(roles_performance_list_absolute_equality_allocation)
		one_assignment_sum_val = roles_performance_list_relative_equality_allocation.sum(axis=0)/cyclingTimes
		record.append(str(one_assignment_sum_val))
		record.append(str(one_assignment_sum_val.sum(axis=1)/4))
		# orig_cplex_result_list = np.array(orig_cplex_result_list)
		# print(orig_cplex_result_list.sum(axis=0)/cyclingTimes)
		record.append("------Re-assignment Result------")
		record.append(str(sum(latest_total_performance_list)/cyclingTimes))
		record.append(str(sum(latest_top_20_percent_list)/cyclingTimes))
		record.append(str(sum(latest_last_20_percent_list)/cyclingTimes))
		latest_hierarchical_list = np.array(latest_hierarchical_list)
		record.append(str(latest_hierarchical_list.sum(axis = 0)/cyclingTimes))
		# print(roles_performance_relative_equality_allocation)
		# print(roles_performance_list_relative_equality_allocation)
		re_assignment_sum_val = roles_performance_list_absolute_equality_allocation.sum(axis=0)/cyclingTimes
		record.append(str(re_assignment_sum_val))
		record.append(str(re_assignment_sum_val.sum(axis=1)/4))
		modified_amount_list = np.array(modified_amount_list)
		# cplex_result_list = np.array(cplex_result_list)
		# print(cplex_result_list.sum(axis = 0)/cyclingTimes)
		record.append(str(modified_amount_list.sum(axis=0)/cyclingTimes))
		recordResult(fileName, record)
		record.append("----------"+str(b_val)+"------------"+str(cyclingTimes)+"---------")