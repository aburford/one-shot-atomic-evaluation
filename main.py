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

def get_gradient(X, y, theta):
	np.set_printoptions(threshold=np.inf)
	m, n = X.shape
	y = np.resize(y, (n, m))
	z = X * y.transpose()
	grad = z @ theta
	grad = 1 / (1 + np.exp(-grad))
	grad = 1 - grad
	grad = np.resize(grad, (n, m))
	grad = grad.transpose() * z
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

def sgd(X, y, batch_size, iters=500, alpha=0.01):
	m, n = X.shape
	theta = np.random.normal(size=n)
	window = 0
	for i in range(iters):
		print('\r%.2f%%' % (i / iters), end='')
		bx, by = X[window:window + batch_size], y[window:window + batch_size]
		if window + batch_size > m:
			bx = np.concatenate((bx, X[:window + batch_size - m]))
			by = np.concatenate((by, Y[:window + batch_size - m]))
		theta -= get_gradient(bx, by, theta)
		window += batch_size
		window %= m
	print()
	return theta

small_n = 300
batch_size = 300

def run_oneshot():
	accs = []
	for i in range(100):
		theta = np.random.normal(size=n)
		grad = get_gradient(X, y, theta)
		biggest = abs(grad).argsort()
		print('selecting %d out of %d features' % (small_n, n))
		Xsub = X.T[biggest < small_n].T
		theta = sgd(Xsub, y, batch_size)
		Xvsub = Xv.T[biggest < small_n].T
		acc = 100 * calc_accuracy(Xvsub, yv, theta)
		print('validation accuracy: %.2f' % (acc))
		accs.append(acc)
		#X = add_noise(X, 50)
	accs = np.array(accs)
	np.savetxt('accs.txt', accs)

	plt.hist(accs, bins=20)
	plt.xlabel('Validation Accuracy (%)')
	plt.ylabel('Count')
	plt.title('Validation Accuracy of Selected Feature Subset')
	plt.savefig('hist.pdf')

def run_mim(X, y):
	# note that I modified this mim function because by default it is O(n**3)
	# but should be O(n**2)
	feats, j_cmi, mify = MIM.mim(X, y, n_selected_features=small_n)
	Xsub = X.T[feats].T
	theta = sgd(Xsub, y, batch_size)
	Xvsub = Xv.T[feats].T
	acc = 100 * calc_accuracy(Xvsub, yv, theta)
	print(acc)

def run_gini(X, y):
	gini_rank = gini_index.gini_index(X, y).argsort()
	Xsub = X.T[gini_rank < small_n].T
	theta = sgd(Xsub, y, batch_size)
	Xvsub = Xv.T[gini_rank < small_n].T
	acc = 100 * calc_accuracy(Xvsub, yv, theta)
	print(acc)

if __name__ == '__main__':
	# X: (batch, pixels)
	# Y: (batch)
	X, y = get_gisette_data()
	Xv, yv = get_gisette_data('valid')
	m, n = X.shape
	print('loaded %d samples' % m)
	#run_mim(X, y)
	run_gini(X, y)
