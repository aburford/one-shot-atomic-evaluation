import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def plot_num(xi, name='plot'):
	plt.figure()
	plot_data = np.reshape(xi, (28, 28))
	plt.imshow(plot_data)
	plt.savefig('%s.pdf' % name)

def get_training_data():
	data = sio.loadmat('mnist.mat')
	ones = data['trainY'] == 0
	zeros = data['trainY'] == 1
	binary = np.concatenate((ones, zeros)).any(axis=0)
	return data['trainX'][binary], data['trainY'][:,binary].squeeze()

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

if __name__ == '__main__':
	# X: (batch, pixels)
	# Y: (batch)
	X, y = get_training_data()
	y = y.astype('int32') * 2 - 1
	batch_size, input_size = X.shape
	theta = np.random.normal(size=input_size)
	theta = np.zeros(input_size)
	grad = get_gradient(X, y, theta)
	plot_num(grad, 'grad')
	biggest = abs(grad).argsort()
	
	#X = add_noise(X, 50)
	
