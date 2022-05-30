# -*- coding: utf-8 -*-
"""
Created on Mon May 30 02:53:22 2022

@author: piton
"""

from numpy import asarray
from numpy import arange
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

from PyQt5.QtWidgets import QFileDialog
 
# objective function
def objective(x):
	return x[0]**2.0
 
# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# evaluate the initial point
	solution_eval = objective(solution)
	# run the hill climb
	solutions = list()
	solutions.append(solution)
	for i in range(n_iterations):
		# take a step
		candidate = solution + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidte_eval = objective(candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# keep track of solutions
			solutions.append(solution)
			# report progress
			print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval, solutions]
 
# seed the pseudorandom number generator
seed(5)
filePath = QFileDialog.getOpenFileName(filter = "Data files (*.txt)")[0]
df = pd.read_csv(filePath,sep=" ",header=None)
df.columns = ["X","Y"]
inputs = np.array(df)
# define range for input
bounds = asarray([[-5.0, 5.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = 0.1
# perform the hill climbing search
kmeans = KMeans(n_clusters=3, random_state=0).fit(inputs)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
best, score, solutions = hillclimbing(objective, bounds, n_iterations, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))
# sample input range uniformly at 0.1 increments
# inputs = arange(bounds[0,0], bounds[0,1], 0.1)
# create a line plot of input vs result
pyplot.plot(inputs, [objective([x]) for x in inputs], '--')
# draw a vertical line at the optimal input
pyplot.axvline(x=[0.0], ls='--', color='red')
# plot the sample as black circles
pyplot.plot(solutions, [objective(x) for x in solutions], 'o', color='black')
pyplot.show()