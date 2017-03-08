import numpy as np
from random import shuffle

# x=['{0:010b}'.format(i) for i in range(1000)]
# shuffle(x)
# data_x=np.array([[int(i) for i in each] for each in x])
# print(data_x.shape)
# ys=np.array([each.count('1') for each in x])
# print(ys.shape)
# # print(data_y)
# data_y=np.array([[0]*10]*1000)
# data_y[range(1000),ys-1]=1
# print(data_x[0],data_y[0])
# print(data_x[1],data_y[1])
# print(data_x[2],data_y[2])
# print(data_x[3],data_y[3])

def gen_data():
		x=['{0:010b}'.format(i) for i in range(1000)]
		shuffle(x)
		data_x=np.array([[int(i) for i in each] for each in x])
		print(data_x.shape)
		ys=np.array([each.count('1') for each in x])
		print(ys.shape)
		# print(data_y)
		data_y=np.array([[0]*10]*1000)
		data_y[range(1000),ys-1]=1
		print(data_x[0],data_y[0])
		return data_x,data_y

def gen_batch(raw_data, batch_size, num_steps):
		raw_x, raw_y = raw_data
		data_x = np.stack(np.vsplit(raw_x,batch_size),axis=2)
		data_y = np.stack(np.vsplit(raw_y,batch_size),axis=2)
		print(data_x.shape)
		# print(data_x)

		for i in range(data_x.shape[0]):
				# print('data_x[i].shape: ',data_x[i].shape)
				yield (data_x[i].T,data_y[i].T)

def gen_bin_seq_epochs(num_epochs, batch_size,num_steps):
		for i in range(num_epochs):
				yield gen_batch(gen_data(), batch_size, num_steps)

# print(gen_bin_seq_epochs(num_epochs=100,batch_size=10,num_steps=10))