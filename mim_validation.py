import argparse
from time import time
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_array
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.statistical_based import gini_index


def get_mnist_data():
	data = sio.loadmat('mnist.mat')
	ones = data['trainY'] == 1
	zeros = data['trainY'] == 0
	binary = np.concatenate((ones, zeros)).any(axis=0)
	
	ones_t = data['testY'] == 1
	zeros_t = data['testY'] == 0
	binary_t = np.concatenate((ones_t, zeros_t)).any(axis=0)
	return data['trainX'][binary]/255, data['trainY'][:,binary].squeeze(), data['testX'][binary_t]/255, data['testY'][:,binary_t].squeeze()


def calc_accuracy(X, y, theta):
	m, n = X.shape
	y_hat = np.sign(X @ theta)
	return sum(y_hat == y.squeeze()) / m


def add_gaussian_noise(X_train, mean, stddev):
	''' 
	INPUT:  (1) 4D numpy array: all raw training image data, of shape 
				(#imgs, #chan, #rows, #cols)
			(2) float: the mean of the Gaussian to sample noise from
			(3) float: the standard deviation of the Gaussian to sample
				noise from. Note that the range of pixel values is
				0-255; choose the standard deviation appropriately. 
	OUTPUT: (1) 4D numpy array: noisy training data, of shape
				(#imgs, #chan, #rows, #cols)
	'''

	n_rows = X_train.shape[0]
	n_cols = X_train.shape[1]
	if stddev == 0:
		noise = np.zeros((n_rows, n_cols))
	else:
		noise = np.random.normal(mean, stddev, 
								 (n_rows, n_cols))
	noisy_X = X_train + noise
	clipped_noisy_X = np.clip(noisy_X, 0., 1.)
	return clipped_noisy_X



def get_gradient(X, y, theta, l1_norm=False):
	np.set_printoptions(threshold=np.inf)
	m, n = X.shape
	y = np.resize(y, (n, m))
	z = X * y.transpose()
	grad = z @ theta
	grad = 1 / (1 + np.exp(-grad))
	grad = 1 - grad
	grad = np.resize(grad, (n, m))
	grad = grad.transpose() * z
	if l1_norm:
		grad += np.sign(theta)
	grad = grad.mean(axis=0)
	return -grad


def select_feats(X, feats):
	if isinstance(X, coo_array):
		select = np.isin(X.col, feats)
		new_shape = (X.shape[0], len(feats))
		cols = X.col[select]
		for i, f in enumerate(feats):
			cols[cols == f] = i
		return coo_array((X.data[select], (X.row[select], cols)), shape=new_shape)
	else:
		return X.T[feats].T



def gd(X, y, iters=3000, alpha=0.1, l1_norm=False):
	
	m, n = X.shape
	theta = np.random.normal(size=n)
	for i in range(iters):

#         print('\r%.2f%%' % (i / iters), end='')
		grad = get_gradient(X, y, theta, l1_norm)
		theta = theta - alpha * grad
#         print(theta[172], theta[186], theta[197], theta[210], theta[224])
		grad_l.append(theta)
		theta_l.append(theta)
#     print()
	return theta




def sgd(X, y, batch_size, iters=500, alpha=0.01, l1_norm=False):
	m, n = X.shape
	theta = np.random.normal(size=n)
	window = 0
	for i in range(iters):
#         if i % 10:
#             print('\r%.2f%%' % (i / iters), end='')
		bx, by = X[window:window + batch_size], y[window:window + batch_size]
		if window + batch_size > m:
			bx = np.concatenate((bx, X[:window + batch_size - m]))
			by = np.concatenate((by, y[:window + batch_size - m]))
		theta -= get_gradient(bx, by, theta, l1_norm)
		window += batch_size
		window %= m
#     print()
	return theta




def run_oneshot(X, y, small_n, iters=1, print_progress=True, train_delay=0):
	m, n = X.shape
	accs = []
	for i in range(iters):
		theta = np.random.normal(size=n)
		if train_delay:
			theta = gd(X, y, iters=train_delay, l1_norm=True)
		grad = abs(get_gradient(X, y, theta))
#         if print_progress:
#             print('\r%.2f%%' % (i / iters), end='')
		biggest = grad.argsort()
		Xsub = select_feats(X, np.where(biggest < small_n)[0])
		Xsub_l.append(Xsub)
		theta = sgd(Xsub, y, batch_size, iters=500, alpha=0.01)
#         theta = gd(Xsub, y)
#         Xvsub = select_feats(Xv, np.where(biggest < small_n)[0])
		Xsub_l.append(Xsub)
		acc = 100 * calc_accuracy(Xsub, y, theta)
		accs.append(acc)
#         print(acc)

#         sys.exit()
#     if print_progress:
#         print()
	return accs




def run_mim(X, y, small_n):
	# note that I modified this mim function because by default it is O(n**3)
	# but should be O(nlogn)
	print("getting mim")
	feats, _, _ = MIM.mim(X, y)
	
	feats = feats[:small_n]
	Xsub = select_feats(X, feats)
#     theta = gd(Xsub, y)
	print("doing sgd")
	theta = sgd(Xsub, y, batch_size, iters=500, alpha=0.01)
#     if isinstance(Xv, coo_array):
#         select = np.isin(Xv.col, feats)
#         new_shape = (Xv.shape[0], small_n)
#         cols = Xv.col[select]
#         for i, f in enumerate(feats):
#             cols[cols == f] = i
#         Xvsub = coo_array((Xv.data[select], (Xv.row[select], cols)), shape=new_shape)
#     else:
#         Xvsub = Xv.T[feats].T
	print("calculating accuracy")
	acc = 100 * calc_accuracy(Xsub, y, theta)
	return acc

def run_gini(X, y, small_n, cache='dorothea-gini'):

	gini_rank = gini_index.gini_index(X, y).argsort()
	Xsub = select_feats(X, np.where(gini_rank < small_n)[0])
	theta = gd(Xsub, y)
#     Xvsub = select_feats(Xv, np.where(gini_rank < small_n)[0])
	acc = 100 * calc_accuracy(Xsub, y, theta)
	return acc



if __name__ == "__main__":

	for i in [32,48,64,80,96,112,128]:

		xtry = mxtr
		ytry = mytr
		mim = run_mim(xtry, ytry, i)
		print(mim)
		print("*********************")