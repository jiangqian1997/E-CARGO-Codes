'''

 * This program is to achieve Equation (8) in the paper named "Agent Evaluation in Deployment of Multi-SUAVs for Communication Recovery".  
 
 * Created by Qian Jiang and Zhiwei He, Jan 7, 2020

'''

def genRandomPoints(ckoutset, R=100, sigma=1):
	"""
	Inputs ------
	ckoutset_arr: 2-D list with shape of num_points x 7
	"""
	# ckoutset_arr = ckoutset_arr.copy() 
	ckoutset_arr = np.array(ckoutset)
	# print(ckoutset_arr)

	num_points, _ = ckoutset_arr.shape
	
	theta, phi = np.pi * np.random.rand(2, num_points)
	random_dir = np.vstack((
							np.multiply(np.sin(phi), np.cos(theta)),
							np.multiply(np.sin(phi), np.sin(theta)),
							np.cos(phi),
							 )).T
	random_len = np.abs(sigma * np.random.randn(num_points)) + R # Truncated gaussian distribution
	# print(random_len)
	random_change =  np.multiply(random_dir, random_len.reshape(-1, 1))

	ckoutset_arr[:, :3] = ckoutset_arr[:, :3] + random_change # original coordinate + random change

	return [ list(i) for i in ckoutset_arr]

def expandPointXYZ(point, currentNum, expand_times=(0, 1, 1), distance=[100, 100, 100], maxlist=[0,0,0]):
	newPoints = []
	for i in range(0,expand_times[0]+1):
		for j in range(0,expand_times[1]+1):
			for k in range(0,expand_times[2]+1):
				# print(currentNum)
				newPoint = [
					point[0] + i*distance[0],
					point[1] + j*distance[1],
					point[2] + k*distance[2],
					# np.random.randint(0, 2),
					# np.random.randint(0, 2),
					point[3],
					point[4],
					currentNum,
					[0, 0],
				]
				newPoints.append(newPoint)
				currentNum += 1
	# newPoints[0][5] = point[5]
	return newPoints, currentNum


def getDivisionTimesList(orig):
	"""
	Inputs ------
	orig: 2-D array with shape of num_points x 3
	"""
	tmp_orig = np.array([i[:3] for i in orig])
	return list(np.max(tmp_orig, 0) - np.min(tmp_orig, 0)), list(np.max(tmp_orig, 0))


# orig: array of point = [x, y, z, typ, corrType, num, errors]
def generateRawData(orig): # dimension -> square division
	divisionTimesList, maxlist = getDivisionTimesList(orig)
	currentNum = 0
	newDataSet = []
	for point in orig:
		newPoints, currentNum = expandPointXYZ(point, currentNum, (2, 0, 1), divisionTimesList, maxlist)
		newDataSet += newPoints
	# newDataSet = newDataSet
	orig = newDataSet.copy()
	orig[0][3]=2
	orig[-1][3]=2
	newDataSet = np.array(newDataSet)
	newDataSet = newDataSet[:, [5, 0, 1, 2, 3, 4]]
	df = pd.DataFrame(newDataSet)
	writer = pd.ExcelWriter('new_dataset.xlsx')
	df.to_excel(writer, 'page_1')
	writer.save()
	return orig