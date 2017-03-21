import numpy as np

# from net_input_X.aritificial_datasets.bin_seq.bin_seq import gen_bin_seq_epochs as gen_epochs
# chars_size=2

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

def gen_batch(raw_data, num_steps,batch_size):
    raw_x, raw_y = raw_data
    data_x = np.stack(np.vsplit(raw_x.reshape((-1,num_steps)),batch_size),axis=2)
    data_y = np.stack(np.vsplit(raw_y.reshape((-1,num_steps)),batch_size),axis=2)
    print('data_x.shape: ',data_x.shape)

    print('data_x[0][0][0:10]: ',data_x[0][0][0:10])
    for i in range(data_x.shape[0]):
        yield (data_x[i].T,data_y[i].T)

def gen_epochs(num_epochs, num_steps,batch_size):
    for i in range(num_epochs):
        yield gen_batch(gen_data(), num_steps,batch_size)


for idx, epoch in enumerate(gen_epochs(num_epochs=1000, num_steps=10, batch_size=100)):
	print(idx,epoch)
	# for X, Y in epoch:
	# 	print(X,Y)

