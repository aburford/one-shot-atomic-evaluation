import argparse
from time import time
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
# install skfeature from here
# https://github.com/jundongl/scikit-feature
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.statistical_based import gini_index

def plot_num(xi, name='plot'):
	plt.figure()
	plot_data = np.reshape(xi, (28, 28))
	plt.imshow(plot_data)
	plt.savefig('%s.pdf' % name)

def get_mnist_data():
	data = sio.loadmat('mnist.mat')
	ones = data['trainY'] == 0
	zeros = data['trainY'] == 1
	binary = np.concatenate((ones, zeros)).any(axis=0)
	return data['trainX'][binary], data['trainY'][:,binary].squeeze()

def get_gisette_data(dtype='train'):
	with open('gisette/gisette_%s.data' % dtype) as f:
		x = [[int(x) for x in line.split()] for line in f.readlines()]
		x = np.array(x)
	with open('gisette/gisette_%s.labels' % dtype) as f:
		y = [[int(y) for y in line.split()] for line in f.readlines()]
		y = np.array(y).squeeze()
	return x, y

def add_noise(X, strength):
	batch_size, input_size = X.shape
	cov = strength * np.identity(input_size)
	spherical_noise = np.random.multivariate_normal(np.ones(input_size), cov, size=batch_size)
	return X + spherical_noise

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

def calc_accuracy(X, y, theta):
	m, n = X.shape
	y_hat = np.sign(X @ theta)
	return sum(y_hat == y.squeeze()) / m

def gd(X, y, iters=500, alpha=0.01):
	m, n = X.shape
	theta = np.random.normal(size=n)
	for i in range(iters):
		print('\r%.2f%%' % (i / iters), end='')
		theta -= alpha * get_gradient(X, y, theta)
	return theta

def sgd(X, y, batch_size, iters=500, alpha=0.01, l1_norm=False):
	m, n = X.shape
	theta = np.random.normal(size=n)
	window = 0
	for i in range(iters):
		if i % 10:
			print('\r%.2f%%' % (i / iters), end='')
		bx, by = X[window:window + batch_size], y[window:window + batch_size]
		if window + batch_size > m:
			bx = np.concatenate((bx, X[:window + batch_size - m]))
			by = np.concatenate((by, Y[:window + batch_size - m]))
		theta -= get_gradient(bx, by, theta, l1_norm)
		window += batch_size
		window %= m
	print()
	return theta

def run_oneshot(X, y, small_n, iters=100, print_progress=True, train_delay=0):
	accs = []
	for i in range(iters):
		theta = np.random.normal(size=n)
		if train_delay:
			theta = sgd(X, y, batch_size, iters=train_delay, l1_norm=True)
		grad = abs(get_gradient(X, y, theta))
		if print_progress:
			print('\r%.2f%%' % (i / iters), end='')
		biggest = grad.argsort()
		Xsub = X.T[biggest < small_n].T
		theta = sgd(Xsub, y, batch_size)
		Xvsub = Xv.T[biggest < small_n].T
		acc = 100 * calc_accuracy(Xvsub, yv, theta)
		accs.append(acc)
	if print_progress:
		print()
	return accs

def run_mim(X, y, small_n, cache='gisette-mim'):
	fn = '%s.txt' % cache
	if os.path.isfile(fn):
		feats = np.loadtxt(fn, dtype='int32')
	else:
		# note that I modified this mim function because by default it is O(n**3)
		# but should be O(n**2)
		feats, _, _ = MIM.mim(X, y)
		np.savetxt(fn, feats)
	feats = feats[:small_n]
	Xsub = X.T[feats].T
	theta = sgd(Xsub, y, batch_size)
	Xvsub = Xv.T[feats].T
	acc = 100 * calc_accuracy(Xvsub, yv, theta)
	return acc

def run_gini(X, y, small_n, cache='gisette-gini'):
	fn = '%s.txt' % cache
	if os.path.isfile(fn):
		gini_rank = np.loadtxt(fn)
	else:
		gini_rank = gini_index.gini_index(X, y).argsort()
		np.savetxt(fn, gini_rank)
	Xsub = X.T[gini_rank < small_n].T
	theta = sgd(Xsub, y, batch_size)
	Xvsub = Xv.T[gini_rank < small_n].T
	acc = 100 * calc_accuracy(Xvsub, yv, theta)
	return acc

def get_histogram_data():
	accs = run_oneshot()
	accs = np.array(accs)
	np.savetxt('accs.txt', accs)
	print('calculating mim')
	acc_mim = run_mim(X, y, small_n)
	print(acc_mim)
	print('calculating gini')
	acc_gini = run_gini(X, y, small_n)
	print(acc_gini)

def make_histogram():
	# replace with actual values
	mim = 93.8
	gini = 93.3
	accs = np.loadtxt('accs.txt')

	plt.hist(accs, bins=20, label='Atomic One-Shot')
	plt.xlabel('Test Accuracy (%)')
	plt.ylabel('Count')
	plt.title('Test Accuracy On Gisette With 300 Features')
	plt.axvline(mim, color='r', label='MIM')
	plt.axvline(gini, color='y', label='Gini Index')
	plt.legend()
	plt.xlim(55, 100)
	plt.savefig('hist.pdf')

small_ns = [2**i for i in list(range(3, 12))] + [5000]

def get_linegraph_data():
	# oneshot, mim, gini
	accs = []
	t1 = time()
	for small_n in small_ns:
		print('testing', small_n)
		acc_mim = run_mim(X, y, small_n)
		t1 = time()
		acc_gini = run_gini(X, y, small_n)
		t1 = time()
		accs_oneshot = np.array(run_oneshot(X, y, small_n, 10, print_progress=False))
		t1 = time()
		accs.append((accs_oneshot.mean(), acc_mim, acc_gini))
	accs = np.array(accs)
	np.savetxt('linegraph-accs.txt', accs)

def make_linegraph():
	accs = np.loadtxt('linegraph-accs.txt')
	print(len(small_ns))
	trunc = 10
	plt.plot(small_ns[:trunc], accs[:,0][:trunc], label='Atomic One-Shot')
	plt.plot(small_ns[:trunc], accs[:,1][:trunc], label='MIM')
	plt.plot(small_ns[:trunc], accs[:,2][:trunc], label='Gini')
	plt.semilogx()
	plt.xticks(ticks=small_ns, labels=small_ns)
	plt.minorticks_off()
	plt.legend()
	plt.ylabel('Test Accuracy (%)')
	plt.xlabel('Size of Feature Set')
	plt.title('Model Accuracy Across Varying Feature Set Size')
	plt.savefig('linegraph.pdf')

def get_boxplot_data():
	num_samples = 10
	iters = 10
	delays = [0] + [2**i for i in range(num_samples-1)]
	accs = np.zeros((num_samples, iters))
	for i, train_delay in enumerate(delays):
		print('train_delay', train_delay)
		accs[i,:] = run_oneshot(X, y, 256, iters=iters, train_delay=train_delay)
	np.savetxt('boxplot.txt', accs)

def make_boxplot():
	num_samples = 10
	delays = [0] + [2**i for i in range(num_samples-1)]
	accs = np.loadtxt('boxplot.txt')
	for i, delay in enumerate(delays):
		if not delay:
			delay = 0.5
		plt.scatter(delay * np.ones(accs.shape[0]), accs[i,:], color='b')
	plt.semilogx()
	plt.xlabel('Training Delay')
	plt.ylim((55, 100))
	plt.ylabel('Test Accuracy (%)')
	plt.title('Test Accuracy As Training Delay Increases')
	plt.xticks(ticks=[0.5] + delays[1:], labels=[0] + delays[1:])
	plt.minorticks_off()
	#plt.legend()
	plt.savefig('boxplot.pdf')
	
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--feature_size', default=300)
	parser.add_argument('--batch_size', default=300)

	args = parser.parse_args()
	small_n = args.feature_size
	batch_size = args.batch_size
	
	# X: (batch, pixels)
	# Y: (batch)

	#print('loading gisette data')
	#X, y = get_gisette_data()
	#Xv, yv = get_gisette_data('valid')
	#m, n = X.shape
	#print('loaded %d samples' % m)

	#compute_300_vals()
	#make_histogram()
	#get_linegraph_data()
	#make_linegraph()
	#get_boxplot_data()
	make_boxplot()
