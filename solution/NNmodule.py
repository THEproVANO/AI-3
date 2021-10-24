import numpy as np
from numpy import linalg as LA
import statistics as st

class Layer(object):
	def linear(self, x):
		return x
	def linear_d(self, x):
		return np.ones(x.shape)
	def relu(self, x):
		return np.where(x < 0, 0, x)
	def relu_d(self, x):
		return np.where(x < 0, 0, 1)
	def sigm(self, x):
		return 1.0/(1.0 + np.exp(-x))
	def sigm_d(self,x):
		return self.sigm(x)*(1 - self.sigm(x))
	def th(self, x):
		return np.tanh(x)
	def th_d(self, x):
		return 1.0 - self.th(x)**2
	def softmax(self, x):
		u = np.exp(x - np.max(x))
		return u/np.sum(u)
	def softmax_d(self, x):
		return x*(1 - x)
	def __init__(self, dim_in, activation='linear'):
		if activation == 'sigm':
			self.f = self.sigm
			self.f_d = self.sigm_d
		elif activation == 'th':
			self.f = self.th
			self.f_d = self.th_d
		elif activation == 'relu':
			self.f = self.relu
			self.f_d = self.relu_d
		elif activation == 'softmax':
			self.f = self.softmax
			self.f_d = self.softmax_d
		else:
			self.f = self.linear
			self.f_d = self.linear_d
		self.dim = dim_in
	def add(self, lr):
		self.weights = np.random.randn(lr.dim, self.dim)
		self.bias = np.random.randn(lr.dim, 1)
	def calculate(self, input):
		return np.dot(self.weights, input) + self.bias

class NN(object):
	def square(self, y_t, y_p):
		return 0.5*(y_t - y_p)**2
	def square_d(self, y_t, y_p):
			return (y_p - y_t)
	def cross_entropy(self, y_t, y_p):
		return -y_t*np.log(y_p)
	def cross_entropy_d(self, y_t, y_p):
		return -y_t/y_p
	def __init__(self):
		self.layers = []
		self.min_loss = float("inf")
		self.min_weights = []
	def add_layer(self, lr):
		self.layers.append(lr)
		if len(self.layers) != 1:
			self.layers[-2].add(lr)
	def answer(self, input):
		if len(self.layers) == 0 or len(self.layers) == 1:
			raise ValueError("bad NN")
		if len(input) != self.layers[0].dim:
			raise ValueError("size input error")
		y = np.array(input)
		self.answers = [0]*(len(self.layers))
		self.answers[0] = np.array(y)
		y.shape = (len(y), 1)
		for i in range(len(self.layers) - 1):
			y = self.layers[i].calculate(y)
			y = self.layers[i + 1].f(y)
			self.answers[i + 1] = y.T[0]
		return y.T[0]

	def fit(self, x, y, loss_in):
		if len(x) != len(y):
			raise ValueError("bad data")
		if loss_in == 'sqr':
			self.loss = self.square
			self.loss_d = self.square_d
		elif loss_in == 'cross_entropy':
			self.loss = self.cross_entropy
			self.loss_d = self.cross_entropy_d
		else:
			raise SyntaxError("no function")
		n = 0.001
		losses = [0]*len(x)
		for data in range(len(x)):
			self.answer(x[data])
			if len(y[data]) != self.layers[-1].dim:
				raise ValueError("bad data")
			losses[data] = LA.norm(self.loss(y[data], self.answers[-1]), 2)
			sigr = self.loss_d(y[data], self.answers[-1])*self.layers[-1].f_d(self.answers[-1])
			for j in range(len(self.layers) - 2, -1, -1):
				lay_sig = np.zeros((self.layers[j].dim))
				for l in range(self.layers[j].dim):
					u = 0
					for r in range(self.layers[j + 1].dim):
						u += self.layers[j].weights[r][l]*sigr[r]
					lay_sig[l] = u*self.layers[j + 1].f_d(self.answers[j][l])
				for l in range(self.layers[j].dim):
					for r in range(self.layers[j + 1].dim):
						self.layers[j].weights[r][l] -= n*sigr[r]*self.answers[j][l]
				for r in range(self.layers[j + 1].dim):
					self.layers[j].bias[r] -= n*sigr[r]
				sigr = lay_sig
		return st.median(losses)

	def test(self, x, y):
		if len(x) != len(y):
			raise ValueError("bad data")
		acc = 0
		for data in range(len(x)):
			self.answer(x[data])
			if len(y[data]) != self.layers[-1].dim:
				raise ValueError("bad data")
			for i in range(len(y[data])):
				if y[data][i] == 1:
					max = float("-inf")
					for j in range(len(y[data])):
						if self.answers[-1][j] > max:
							max = self.answers[-1][j]
							index = j
					if index == i:
						acc += 1
					break
		print("accuracy: ", float(acc)/len(x))