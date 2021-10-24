import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from NNmodule import NN

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.resize((len(x_train), 28*28))
#x_train = x_train[0:20000]
x_train = np.array(x_train, dtype=float)
#y_train = y_train[0:20000]

Y = [0]*len(y_train)
for i in range(len(y_train)):
	ans = y_train[i]
	Y[i] = np.zeros((10,))
	Y[i][ans] = 1

for i in range(len(x_train)):
	for j in range(28*28):
		x_train[i][j] /= 255.0
X = x_train

model = NN()
model.add_layer(Layer(28*28))
model.add_layer(Layer(20, 'sigm'))
model.add_layer(Layer(10, 'softmax'))

start = 0
step = 1000
end = step
points = []
for i in range(len(X)//step):
	points.append(model.fit(X[start:end], Y[start:end], 'sqr'))
	start += step
	end += step

model.test(X[0:1000], Y[0:1000])

plt.plot(points)
plt.title('Loss')
plt.show()